use std::{
    borrow::Borrow,
    collections::{BTreeMap, BTreeSet},
    fs,
    io::{Read, Write},
};

use zkpoly_common::{
    arith::{
        Arith, ArithBinOp, ArithGraph, ArithUnrOp, BinOp, FusedType, Mutability, Operation, SpOp,
        UnrOp,
    },
    get_project_root::get_project_root,
    heap::UsizeId,
    typ::PolyType,
};

use super::FusedOp;
use super::FIELD_NAME;

const TMP_PREFIX: &str = "tmp";
const SUB_FUNC_NAME: &str = "part";
const HEADER_SUFFIX: &str = "_header.cuh";
const TABS: &str = "    ";
const WRAPPER_SUFFIX: &str = "_wrapper";

impl<OuterId: UsizeId, InnerId: UsizeId + 'static> FusedOp<OuterId, InnerId> {
    pub fn num_scalars(&self) -> (usize, usize) {
        let mut num_scalars = 0;
        let mut num_mut_scalars = 0;
        for (typ, _) in self.vars.iter() {
            if *typ == FusedType::Scalar {
                num_scalars += 1;
            }
        }
        for (typ, _) in self.mut_vars.iter() {
            if *typ == FusedType::Scalar {
                num_mut_scalars += 1;
            }
        }
        (num_scalars, num_mut_scalars)
    }

    pub fn new(graph: ArithGraph<OuterId, InnerId>, name: String, limbs: usize) -> Self {
        let (vars, mut_vars) = graph.gen_var_lists();
        let (schedule, live_ts, regs) = graph.schedule();
        let partition = graph.partition(&schedule, 1024); // magic number for compile time

        // now we have to generate the inputs and outputs for each partition
        let mut inputs = vec![BTreeSet::new(); partition.len()];
        let mut outputs = vec![BTreeSet::new(); partition.len()];

        let mut past_vars = BTreeSet::new();
        for i in 0..(partition.len() - 1) {
            let start = partition[i];
            let end = partition[i + 1];
            let mut cur_vars = BTreeSet::new();
            for j in start..end {
                let cur_node = schedule[j];
                let vertex = graph.g.vertex(cur_node);
                cur_vars.insert(cur_node);
                match &vertex.op {
                    Operation::Arith(arith) => {
                        for depend in arith.uses() {
                            if past_vars.contains(&depend) {
                                // this is an input
                                inputs[i].insert(depend);
                            }
                        }
                        if live_ts[cur_node.clone().into()] as usize >= end {
                            // this is an output
                            outputs[i].insert(cur_node);
                        }
                    }
                    Operation::Output { store_node, .. } => {
                        if past_vars.contains(store_node) {
                            // this is an input
                            inputs[i].insert(*store_node);
                        }
                    }
                    Operation::Input { .. } => {
                        if live_ts[cur_node.clone().into()] as usize >= end {
                            // this is an output
                            outputs[i].insert(cur_node);
                        }
                    }
                    Operation::Todo => unreachable!("todo should be handled earlier"),
                }
            }
            past_vars.extend(cur_vars);
        }

        let u32_regs = regs * limbs;
        let launch_bounds = if u32_regs < 128 {
            1024
        } else if u32_regs < 256 {
            512
        } else {
            256
        };

        Self {
            graph,
            name,
            vars,
            mut_vars,
            schedule,
            partition,
            launch_bounds,
            inputs,
            outputs,
        }
    }

    fn compare_or_write(&self, path: &str, s: &str) {
        if let Ok(mut f) = fs::File::open(path) {
            let mut old = String::new();
            f.read_to_string(&mut old).unwrap();
            if s == old {
                return;
            }
        }

        let mut f = fs::File::create(path).unwrap();
        f.write_all(s.as_bytes()).unwrap();
    }

    pub fn gen(&self, head_annotation: impl Borrow<str>) {
        let header = format!("{}\n{}", head_annotation.borrow(), self.gen_header());
        let kernels = self.gen_kernel();
        let wrapper = self.gen_wrapper();

        let project_root = get_project_root();
        let base_path = project_root + "/core/src/fused_kernels/src/" + self.name.as_str();

        // header file
        let header_path = base_path.clone() + HEADER_SUFFIX;
        self.compare_or_write(&header_path, &header);

        let max_regcount = 1024 * 64 / self.launch_bounds; // WARNING: this is ok for ampere, but previous archs may not

        // wrapper file and kernel files has a suffix regarding the launch bounds
        // this is used to inform xmake to set maxregcount
        // because we find sometimes the auto maxregcount is not very efficient
        // wrapper file
        let wrapper_path = base_path.clone() + WRAPPER_SUFFIX + &format!("_regs{}.cu", max_regcount);
        self.compare_or_write(&wrapper_path, &wrapper);

        for (id, kernel) in kernels.iter().enumerate() {
            let kernel_path = base_path.clone() + &format!("_{SUB_FUNC_NAME}{id}_regs{}.cu", max_regcount);
            self.compare_or_write(&kernel_path, kernel);
        }
    }

    fn gen_device_func_args(&self, id: usize, typename: &str) -> String {
        let mut func_sig = String::new();
        func_sig.push_str("(\n");
        func_sig.push_str(&format!(
            "{TABS}ConstPolyPtr const* vars, PolyPtr const* mut_vars, unsigned long long idx, bool is_first",
        ));

        // inputs
        for input_id in self.inputs[id].iter() {
            let input_id: usize = input_id.clone().into();
            func_sig += &format!(",\n{TABS}const {typename} {TMP_PREFIX}{input_id}");
        }

        // outputs
        for output_id in self.outputs[id].iter() {
            let output_id: usize = output_id.clone().into();
            func_sig += &format!(",\n{TABS}{typename} &{TMP_PREFIX}{output_id}");
        }
        func_sig.push_str("\n)");
        func_sig
    }

    fn gen_header(&self) -> String {
        let mut header = String::new();
        header.push_str("#pragma once\n");
        header.push_str("#include \"../../common/mont/src/field_impls.cuh\"\n");
        header.push_str("#include \"../../common/iter/src/iter.cuh\"\n");
        header.push_str("#include \"../../common/error/src/check.cuh\"\n");
        header.push_str("#include <cuda_runtime.h>\n");
        header.push_str("using iter::SliceIterator;\n");
        header.push_str("using mont::u32;\n");
        header.push_str("using iter::make_slice_iter;\n");

        header.push_str("namespace detail {\n");

        // generate the kernel signatures
        for id in 0..(self.partition.len() - 1) {
            header.push_str(&format!(
                "template <typename Field>\n__device__ void {}_{SUB_FUNC_NAME}{id}",
                self.name
            ));
            header.push_str(&self.gen_device_func_args(id, "Field"));
            header.push_str(";\n\n");
        }
        header.push_str("}\n");

        header
    }

    fn gen_wrapper(&self) -> String {
        let mut wrapper = String::new();

        wrapper += &format!("#include \"{}{HEADER_SUFFIX}\"\n\n", self.name);

        wrapper.push_str("namespace detail {\n");

        // generate kernel signature
        wrapper.push_str("template <typename Field>\n");
        wrapper += "__global__ void ";
        wrapper.push_str(&self.name);
        wrapper.push_str(
            "(ConstPolyPtr const* vars, PolyPtr const* mut_vars, unsigned long long len, bool is_first) {\n",
        );

        wrapper +=
            &format!("{TABS}unsigned long long idx = blockIdx.x * blockDim.x + threadIdx.x;\n");
        wrapper += &format!("{TABS}if (idx >= len) return;\n");

        // call the partitioned kernels
        for id in 0..(self.partition.len() - 1) {
            // create output variables
            for (output_rank, output_id) in self.outputs[id].iter().enumerate() {
                let output_id: usize = output_id.clone().into();
                if output_rank == 0 {
                    wrapper += &format!("{TABS}Field ");
                }
                if output_rank + 1 == self.outputs[id].len() {
                    wrapper += &format!("{TMP_PREFIX}{output_id};\n");
                } else {
                    wrapper += &format!("{TMP_PREFIX}{output_id}, ");
                }
            }

            wrapper += &format!(
                "{TABS}{}_{SUB_FUNC_NAME}{id}<Field>(\n{TABS}{TABS}vars, mut_vars, idx, is_first",
                self.name
            );
            // inputs
            for input_id in self.inputs[id].iter() {
                let input_id: usize = input_id.clone().into();
                wrapper += &format!(",\n{TABS}{TABS}{TMP_PREFIX}{input_id}");
            }
            // outputs
            for output_id in self.outputs[id].iter() {
                let output_id: usize = output_id.clone().into();
                wrapper += &format!(",\n{TABS}{TABS}{TMP_PREFIX}{output_id}");
            }
            wrapper.push_str(&format!("\n{TABS});\n"));
        }
        wrapper.push_str("}\n");
        wrapper.push_str("}\n");

        // outer C signature
        wrapper.push_str("extern \"C\" cudaError_t ");
        wrapper.push_str(&self.name);
        wrapper.push_str(
            "(ConstPolyPtr const* vars, PolyPtr const* mut_vars, unsigned long long len, bool is_first, cudaStream_t stream) {\n",
        );

        wrapper.push_str("uint block_size = 256;\n");
        wrapper.push_str("uint grid_size = (len + block_size - 1) / block_size;\n");
        wrapper.push_str("detail::");
        wrapper.push_str(&self.name);
        wrapper.push_str(&format!("<{FIELD_NAME}>"));
        wrapper.push_str(
            " <<< grid_size, block_size, 0, stream >>> (vars, mut_vars, len, is_first);\n",
        );

        wrapper.push_str("return cudaGetLastError();\n");
        wrapper.push_str("}\n");
        wrapper
    }

    fn gen_kernel(&self) -> Vec<String> {
        let mut kernels = Vec::new();

        // prepare the mapping from the input/output to the kernel arguments
        let var_mapping = self
            .vars
            .iter()
            .enumerate()
            .map(|(i, (_, id))| {
                let id: usize = id.clone().into();
                (id, i)
            })
            .collect::<BTreeMap<_, _>>();
        let mut_var_mapping = {
            let mut x = self
                .mut_vars
                .iter()
                .enumerate()
                .map(|(i, (_, id))| {
                    let id: usize = id.clone().into();
                    (id, i)
                })
                .collect::<BTreeMap<_, _>>();

            self.graph.outputs.iter().for_each(|oi| {
                let (_, _, _, in_node) = self.graph.g.vertex(*oi).op.unwrap_output();
                if let Some(in_node) = in_node {
                    if self.graph.g.vertex(*in_node).op.unwrap_input_mutability() == Mutability::Mut
                    {
                        let in_node_usize: usize = in_node.clone().into();
                        let oi_usize: usize = oi.clone().into();
                        x.insert(in_node_usize, x[&oi_usize]);
                    }
                }
            });

            x
        };

        for id in 0..(self.partition.len() - 1) {
            let mut kernel = String::new();
            kernel.push_str(&format!("#include \"{}{HEADER_SUFFIX}\"\n", self.name));
            kernel.push_str("namespace detail {\n");

            // generate the kernel signature
            kernel.push_str(&format!(
                "template <typename Field>\n__device__ void {}_{SUB_FUNC_NAME}{id}",
                self.name
            ));
            kernel.push_str(&self.gen_device_func_args(id, "Field"));
            kernel.push_str(" {\n");

            let start = self.partition[id];
            let end = self.partition[id + 1];

            for step in start..end {
                let head = self.schedule[step];
                let vertex = self.graph.g.vertex(head.clone());
                let store_target = if self.outputs[id].contains(&head) {
                    format!("{TMP_PREFIX}{}", head.clone().into())
                } else {
                    format!("auto {TMP_PREFIX}{}", head.clone().into())
                };
                match &vertex.op {
                    Operation::Output {
                        typ, store_node, ..
                    } => {
                        let src: usize = store_node.clone().into();
                        let id: usize = head.clone().into(); // self.outputs_i2o[&head].clone().into();
                        match typ {
                            FusedType::Scalar => {
                                kernel += &format!("*reinterpret_cast<Field*>(mut_vars[{}].ptr) = {TMP_PREFIX}{src};\n", mut_var_mapping[&id]);
                            }
                            FusedType::ScalarArray => {
                                kernel += &format!("make_slice_iter<Field>(mut_vars[{}])[idx] = {TMP_PREFIX}{src};\n", mut_var_mapping[&id]);
                            }
                        }
                    }
                    Operation::Input { typ, .. } => {
                        let id = head.clone().into(); // outer_id.clone().into();
                        let (map_id, mutability) = if var_mapping.contains_key(&id) {
                            (var_mapping[&id], "vars")
                        } else {
                            (mut_var_mapping[&id], "mut_vars")
                        };
                        match typ {
                            FusedType::Scalar => {
                                kernel += &format!("{store_target} = *reinterpret_cast<const Field*>({}[{}].ptr);\n", mutability, map_id);
                            }
                            FusedType::ScalarArray => {
                                kernel += &format!(
                                    "{store_target} = make_slice_iter<Field>({}[{}])[idx];\n",
                                    mutability, map_id
                                );
                            }
                        }
                    }
                    Operation::Arith(arith) => match arith {
                        Arith::Bin(op, lhs, rhs) => {
                            let lhs: usize = lhs.clone().into();
                            let rhs: usize = rhs.clone().into();
                            match op {
                                BinOp::Pp(op) => match op {
                                    ArithBinOp::Add => {
                                        kernel += &format!(
                                        "{store_target} = {TMP_PREFIX}{lhs} + {TMP_PREFIX}{rhs};\n"
                                    )
                                    }
                                    ArithBinOp::Sub => {
                                        kernel += &format!(
                                        "{store_target} = {TMP_PREFIX}{lhs} - {TMP_PREFIX}{rhs};\n"
                                    )
                                    }
                                    ArithBinOp::Mul => {
                                        kernel += &format!(
                                        "{store_target} = {TMP_PREFIX}{lhs} * {TMP_PREFIX}{rhs};\n"
                                    )
                                    }
                                    ArithBinOp::Div => {
                                        unreachable!("division should be handled in batched invert")
                                    }
                                },
                                BinOp::Ss(op) => match op {
                                    ArithBinOp::Add => {
                                        kernel += &format!(
                                        "{store_target} = {TMP_PREFIX}{lhs} + {TMP_PREFIX}{rhs};\n"
                                    )
                                    }
                                    ArithBinOp::Sub => {
                                        kernel += &format!(
                                        "{store_target} = {TMP_PREFIX}{lhs} - {TMP_PREFIX}{rhs};\n"
                                    )
                                    }
                                    ArithBinOp::Mul => {
                                        kernel += &format!(
                                        "{store_target} = {TMP_PREFIX}{lhs} * {TMP_PREFIX}{rhs};\n"
                                    )
                                    }
                                    ArithBinOp::Div => {
                                        unreachable!("division should be handled in scalar invert")
                                    }
                                },
                                BinOp::Sp(op) => match op {
                                    SpOp::Add => {
                                        let stmt = match self.graph.poly_repr {
                                            PolyType::Coef => {
                                                format!("{store_target} = (idx == 0 && is_first) ? {TMP_PREFIX}{lhs} + {TMP_PREFIX}{rhs} : {TMP_PREFIX}{rhs};\n")
                                            }
                                            PolyType::Lagrange => {
                                                format!("{store_target} = {TMP_PREFIX}{lhs} + {TMP_PREFIX}{rhs};\n")
                                            }
                                        };
                                        kernel += &stmt;
                                    }
                                    SpOp::Sub => {
                                        let stmt = match self.graph.poly_repr {
                                            PolyType::Coef => {
                                                format!("{store_target} = (idx == 0 && is_first) ? {TMP_PREFIX}{lhs} - {TMP_PREFIX}{rhs} : {TMP_PREFIX}{rhs}.neg();\n")
                                            }
                                            PolyType::Lagrange => {
                                                format!("{store_target} = {TMP_PREFIX}{lhs} - {TMP_PREFIX}{rhs};\n")
                                            }
                                        };
                                        kernel += &stmt;
                                    }
                                    SpOp::SubBy => {
                                        let stmt = match self.graph.poly_repr {
                                            PolyType::Coef => {
                                                format!("{store_target} = (idx == 0 && is_first) ? {TMP_PREFIX}{rhs} - {TMP_PREFIX}{lhs} : {TMP_PREFIX}{rhs};\n")
                                            }
                                            PolyType::Lagrange => {
                                                format!("{store_target} = {TMP_PREFIX}{rhs} - {TMP_PREFIX}{lhs};\n")
                                            }
                                        };
                                        kernel += &stmt;
                                    }
                                    SpOp::Mul => {
                                        kernel += &format!(
                                        "{store_target} = {TMP_PREFIX}{lhs} * {TMP_PREFIX}{rhs};\n"
                                    )
                                    }
                                    SpOp::Div => {
                                        unreachable!("division should be handled in type2")
                                    }
                                    SpOp::DivBy => {
                                        unreachable!("division should be handled in type2")
                                    }
                                },
                            }
                        }
                        Arith::Unr(op, arg) => {
                            let arg: usize = arg.clone().into();
                            match op {
                                UnrOp::P(ArithUnrOp::Neg) => {
                                    kernel +=
                                        &format!("{store_target} = {TMP_PREFIX}{arg}.neg();\n");
                                }
                                UnrOp::P(ArithUnrOp::Inv) => {
                                    unreachable!("invert poly should be handled in batche invert")
                                }
                                UnrOp::P(ArithUnrOp::Pow(power)) => {
                                    kernel += &format!(
                                        "{store_target} = {TMP_PREFIX}{arg}.pow({power});\n"
                                    );
                                }
                                UnrOp::S(ArithUnrOp::Neg) => {
                                    kernel +=
                                        &format!("{store_target} = {TMP_PREFIX}{arg}.neg();\n");
                                }
                                UnrOp::S(ArithUnrOp::Inv) => {
                                    unreachable!("invert scalar should be handled in scalar invert")
                                }
                                UnrOp::S(ArithUnrOp::Pow(_)) => {
                                    unreachable!("power scalar should be handled in scalar power")
                                }
                            }
                        }
                    },
                    Operation::Todo => unreachable!("todo should be handled earlier"),
                }
            }

            kernel.push_str("}\n");

            // explicitly instantiate the template
            kernel += &format!(
                "template __device__ void {}_{SUB_FUNC_NAME}{id}<{FIELD_NAME}>",
                self.name
            );
            kernel.push_str(&self.gen_device_func_args(id, FIELD_NAME));
            kernel.push_str(";\n");

            kernel.push_str("}\n");
            kernels.push(kernel);
        }
        kernels
    }
}
