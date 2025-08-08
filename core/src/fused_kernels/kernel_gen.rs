use std::{
    borrow::Borrow,
    collections::{BTreeMap, VecDeque},
    fs,
    io::{Read, Write},
    path::{Path, PathBuf},
};

use zkpoly_common::{
    arith::{
        Arith, ArithBinOp, ArithGraph, ArithUnrOp, BinOp, FusedType, Mutability, Operation, SpOp,
        UnrOp,
    },
    belady_allocator::{BeladyAllocator, PageId, PageLocation, SpillPageManger},
    get_project_root::get_project_root,
    heap::{Heap, UsizeId},
    typ::PolyType,
};
use zkpoly_cuda_api::device_info::get_num_sms;

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

    pub fn new(mut graph: ArithGraph<OuterId, InnerId>, name: String, limbs: usize) -> Self {
        let (vars, mut_vars) = graph.gen_var_lists();
        let (schedule, _, regs) = graph.schedule();
        let partition = graph.partition(&schedule, 1024); // magic number for compile time

        // prepare the mapping from the input/output to the kernel arguments
        let var_mapping = vars
            .iter()
            .enumerate()
            .map(|(i, (_, id))| {
                let id: usize = id.clone().into();
                (id, i)
            })
            .collect::<BTreeMap<_, _>>();
        let mut_var_mapping = {
            let mut x = mut_vars
                .iter()
                .enumerate()
                .map(|(i, (_, id))| {
                    let id: usize = id.clone().into();
                    (id, i)
                })
                .collect::<BTreeMap<_, _>>();

            graph.outputs.iter().for_each(|oi| {
                let (_, _, _, in_node) = graph.g.vertex(*oi).op.unwrap_output();
                if let Some(in_node) = in_node {
                    if graph.g.vertex(*in_node).op.unwrap_input_mutability() == Mutability::Mut {
                        let in_node_usize: usize = in_node.clone().into();
                        let oi_usize: usize = oi.clone().into();
                        x.insert(in_node_usize, x[&oi_usize]);
                    }
                }
            });

            x
        };

        // eliminate all in_node in the output, as this is not needed
        // and may cause trouble in the kernel generation
        graph.outputs.iter().for_each(|oid| {
            if let Operation::Output { in_node, .. } = &mut graph.g.vertex_mut(*oid).op {
                *in_node = None;
            } else {
                unreachable!("output should be output")
            }
        });

        let u32_regs = regs * limbs;
        let reg_limit = if u32_regs < 128 {
            64
        } else if u32_regs < 256 {
            128
        } else {
            256
        };

        let field_regs = reg_limit / limbs as u32 - 1; // -1 saved for indexs and other things

        assert!(field_regs > 2);

        Self {
            graph,
            name,
            vars,
            mut_vars,
            schedule,
            partition,
            reg_limit,
            field_regs,
            var_mapping,
            mut_var_mapping,
            limbs,
        }
    }

    fn compare_or_write(&self, path: &impl AsRef<Path>, s: &str) {
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

    pub fn get_temp_buffer_size(&self) -> usize {
        let (_, _, spilled_regs) = self.gen_kernel();
        let block_size = 1024 * 64 / self.reg_limit;
        let grid_size = get_num_sms(0);
        let size = spilled_regs
            * block_size as usize
            * grid_size as usize
            * std::mem::size_of::<u32>()
            * self.limbs;
        size
    }

    pub fn gen(&self, head_annotation: impl Borrow<str>, target_path: PathBuf) {
        let (kernels, used_regs, spilled_regs) = self.gen_kernel();
        let header = format!(
            "{}\n{}",
            head_annotation.borrow(),
            self.gen_header(used_regs)
        );
        let wrapper = self.gen_wrapper(used_regs, spilled_regs);

        let base_path = {
            let xmake_template_path =
                get_project_root() + "/core/src/fused_kernels/xmake.lua.template";
            let mut xmake_template = String::new();
            fs::File::open(xmake_template_path)
                .unwrap()
                .read_to_string(&mut xmake_template)
                .unwrap();
            let xmake_target_path = target_path.clone().join("xmake.lua");
            self.compare_or_write(&xmake_target_path, &xmake_template);
            target_path
        };

        // header file
        let header_path = base_path.clone().join(self.name.clone() + HEADER_SUFFIX);
        self.compare_or_write(&header_path, &header);

        // wrapper file and kernel files has a suffix regarding the launch bounds
        // this is used to inform xmake to set maxregcount
        // because we find sometimes the auto maxregcount is not very efficient
        // wrapper file
        let wrapper_path = base_path.join(format!(
            "{}{}_regs{}.cu",
            self.name, WRAPPER_SUFFIX, self.reg_limit
        ));
        self.compare_or_write(&wrapper_path, &wrapper);

        for (id, kernel) in kernels.iter().enumerate() {
            let kernel_path = base_path.join(format!(
                "{}_{SUB_FUNC_NAME}{id}_regs{}.cu",
                self.name, self.reg_limit
            ));
            self.compare_or_write(&kernel_path, kernel);
        }
    }

    fn gen_device_func_args(&self, typename: &str, used_regs: usize) -> String {
        let mut func_sig = String::new();
        func_sig.push_str("(\n");
        func_sig.push_str(&format!(
            "{TABS}ConstPolyPtr const* vars, PolyPtr const* mut_vars, unsigned long long idx, bool is_first, {typename}* local_buffer",
        ));

        // regs
        for i in 0..used_regs {
            func_sig += &format!(",\n{TABS}{typename} &{TMP_PREFIX}{i}");
        }

        func_sig.push_str("\n)");
        func_sig
    }

    fn gen_header(&self, used_regs: usize) -> String {
        let mut header = String::new();
        header.push_str("#pragma once\n");
        let project_root = get_project_root();
        header.push_str(&format!(
            "#include \"{project_root}/core/src/common/mont/src/field_impls.cuh\"\n"
        ));
        header.push_str(&format!(
            "#include \"{project_root}/core/src/common/iter/src/iter.cuh\"\n"
        ));
        header.push_str(&format!(
            "#include \"{project_root}/core/src/common/error/src/check.cuh\"\n"
        ));
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
            header.push_str(&self.gen_device_func_args("Field", used_regs));
            header.push_str(";\n\n");
        }
        header.push_str("}\n");

        header
    }

    fn gen_wrapper(&self, used_regs: usize, spilled_regs: usize) -> String {
        let mut wrapper = String::new();

        wrapper += &format!("#include \"{}{HEADER_SUFFIX}\"\n\n", self.name);

        wrapper.push_str("namespace detail {\n");

        // generate kernel signature
        wrapper.push_str("template <typename Field>\n");
        wrapper += "__global__ void ";
        wrapper.push_str(&self.name);
        wrapper.push_str(
            "(ConstPolyPtr const* vars, PolyPtr const* mut_vars, unsigned long long len, bool is_first, void* local_buffer_ptr) {\n",
        );

        wrapper += &format!("{TABS}u32 thread_id = blockIdx.x * blockDim.x + threadIdx.x;\n");
        wrapper += &format!("{TABS}auto local_buffer = reinterpret_cast<Field*>(local_buffer_ptr) + thread_id * {spilled_regs};\n");

        wrapper += &format!("{TABS}u32 total_threads = blockDim.x * gridDim.x;\n");
        wrapper +=
            &format!("{TABS}for (u32 idx = thread_id; idx < len; idx += total_threads) {{\n");

        for i in 0..used_regs {
            wrapper += &format!("{TABS}{TABS}Field {TMP_PREFIX}{i};\n");
        }
        // call the partitioned kernels
        for id in 0..(self.partition.len() - 1) {
            wrapper += &format!(
                "{TABS}{}_{SUB_FUNC_NAME}{id}<Field>(\n{TABS}{TABS}vars, mut_vars, idx, is_first, local_buffer",
                self.name
            );

            for i in 0..used_regs {
                wrapper += &format!(",\n{TABS}{TABS}{TMP_PREFIX}{i}");
            }

            wrapper.push_str(&format!("\n{TABS});\n"));
        }
        wrapper.push_str(&format!("{TABS}}}\n"));
        wrapper.push_str("}\n");
        wrapper.push_str("}\n");

        // outer C signature
        wrapper.push_str("extern \"C\" cudaError_t ");
        wrapper.push_str(&self.name);
        wrapper.push_str(
            "(ConstPolyPtr const* vars, PolyPtr const* mut_vars, unsigned long long len, bool is_first, void* local_buffer, cudaStream_t stream) {\n",
        );

        wrapper += &format!("auto func = detail::{}<{}>;\n", self.name, FIELD_NAME);

        // we don't use shared memory
        wrapper += &format!("CUDA_CHECK(cudaFuncSetAttribute(func, cudaFuncAttributePreferredSharedMemoryCarveout, cudaSharedmemCarveoutMaxL1));\n");
        let block_size = 1024 * 64 / self.reg_limit;
        let grid_size = get_num_sms(0);
        wrapper.push_str(&format!("uint block_size = {block_size};\n"));
        wrapper.push_str(&format!("uint grid_size = {grid_size};\n"));

        wrapper.push_str(
            "func<<< grid_size, block_size, 0, stream >>> (vars, mut_vars, len, is_first, local_buffer);\n",
        );
        wrapper += "CUDA_CHECK(cudaGetLastError());\n";

        wrapper.push_str("return cudaSuccess;\n");
        wrapper.push_str("}\n");
        wrapper
    }

    fn get_next_use(&self) -> Heap<InnerId, VecDeque<usize>> {
        let mut next_use = Heap::new();

        for _ in 0..self.graph.g.order() {
            next_use.push(VecDeque::new());
        }

        for step in 0..self.schedule.len() {
            let cur = self.schedule[step];
            let vertex = self.graph.g.vertex(cur.clone());
            for from in vertex.uses() {
                next_use[from].push_back(step);
            }
        }
        next_use
    }

    fn gen_kernel(&self) -> (Vec<String>, usize, usize) {
        let mut kernel_bodies = Vec::new();

        // analysis next uses
        let mut next_use = self.get_next_use();

        let mut spill_manager = SpillPageManger::new();
        let mut arithid2pageid = BTreeMap::new();
        let mut allocator = BeladyAllocator::new(1, self.field_regs as usize);

        for id in 0..(self.partition.len() - 1) {
            let mut kernel_body = String::new();

            let start = self.partition[id];
            let end = self.partition[id + 1];

            for step in start..end {
                let cur = self.schedule[step];
                let vertex = self.graph.g.vertex(cur.clone());

                // restore the registers, we need to get arith2reg because they may be freed later
                let mut arith2reg = vertex.uses().fold(BTreeMap::new(), |mut acc, src| {
                    let page_id: PageId = arithid2pageid[&src];
                    let reg_id = if let PageLocation::InMemory(reg_id) =
                        allocator.get_page_location(page_id.clone()).unwrap()
                    {
                        reg_id
                    } else {
                        let (reg_id, evicted) = allocator.restore(page_id); // next use will be updated later
                        if let Some(evict_id) = evicted {
                            let spill_pos = spill_manager.allocate(evict_id.0);
                            // emit the evict code
                            kernel_body += &format!(
                                "{TABS}local_buffer[{spill_pos}] = {TMP_PREFIX}{};\n",
                                evict_id.1
                            );
                        }
                        // emit the restore code
                        kernel_body += &format!(
                            "{TABS}{TMP_PREFIX}{reg_id} = local_buffer[{}];\n",
                            spill_manager.get_spill_pos(page_id.clone()).unwrap()
                        );

                        // remove from the spill manager
                        spill_manager.deallocate(page_id);
                        reg_id
                    };
                    acc.insert(src, reg_id);
                    acc
                });

                // update the next use
                // this need to be done separately, to aviod evicting the using registers
                vertex.uses().for_each(|src| {
                    let page_id: PageId = arithid2pageid[&src];
                    let next_use = next_use[src].pop_front();
                    if let Some(next_use) = next_use {
                        assert!(next_use >= step);
                        allocator.update_next_use(page_id, next_use as u64);
                    } else {
                        allocator.deallocate(page_id);
                    }
                });

                // now we can allocate the target register
                match vertex.op {
                    Operation::Output { .. } | Operation::Todo => {}
                    _ => {
                        let next_use = next_use[cur].pop_front().unwrap(); // can't be empty, otherwise this is a dead code
                        let (page_id, evicted_pages) = allocator.allocate(1, next_use as u64);
                        assert_eq!(page_id.len(), 1);
                        assert!(evicted_pages.len() <= 1);
                        if evicted_pages.len() > 0 {
                            let evict_id = evicted_pages[0];
                            let spill_pos = spill_manager.allocate(evict_id.0);
                            // emit the evict code
                            kernel_body += &format!(
                                "{TABS}local_buffer[{spill_pos}] = {TMP_PREFIX}{};\n",
                                evict_id.1
                            );
                        }
                        arithid2pageid.insert(cur.clone(), page_id[0].clone());
                        arith2reg.insert(
                            cur,
                            allocator
                                .get_page_location(page_id[0].clone())
                                .unwrap()
                                .unwrap_in_memory(),
                        );
                    }
                }

                match &vertex.op {
                    Operation::Output {
                        typ, store_node, ..
                    } => {
                        let src = arith2reg[store_node];
                        let id: usize = cur.clone().into();
                        match typ {
                            FusedType::Scalar => {
                                kernel_body += &format!("{TABS}*reinterpret_cast<Field*>(mut_vars[{}].ptr) = {TMP_PREFIX}{src};\n", self.mut_var_mapping[&id]);
                            }
                            FusedType::ScalarArray => {
                                kernel_body += &format!("{TABS}make_slice_iter<Field>(mut_vars[{}])[idx] = {TMP_PREFIX}{src};\n", self.mut_var_mapping[&id]);
                            }
                        }
                    }
                    Operation::Input { typ, .. } => {
                        let id = cur.clone().into();
                        let (map_id, mutability) = if self.var_mapping.contains_key(&id) {
                            (self.var_mapping[&id], "vars")
                        } else {
                            (self.mut_var_mapping[&id], "mut_vars")
                        };
                        let store_target = arith2reg[&cur];
                        match typ {
                            FusedType::Scalar => {
                                kernel_body += &format!("{TABS}{TMP_PREFIX}{store_target} = *reinterpret_cast<const Field*>({}[{}].ptr);\n", mutability, map_id);
                            }
                            FusedType::ScalarArray => {
                                kernel_body += &format!(
                                    "{TABS}{TMP_PREFIX}{store_target} = make_slice_iter<Field>({}[{}])[idx];\n",
                                    mutability, map_id
                                );
                            }
                        }
                    }
                    Operation::Arith(arith) => match arith {
                        Arith::Bin(op, lhs, rhs) => {
                            let lhs = arith2reg[lhs];
                            let rhs = arith2reg[rhs];
                            let store_target = arith2reg[&cur];
                            match op {
                                BinOp::Pp(op) => match op {
                                    ArithBinOp::Add => {
                                        kernel_body += &format!(
                                        "{TABS}{TMP_PREFIX}{store_target} = {TMP_PREFIX}{lhs} + {TMP_PREFIX}{rhs};\n"
                                    )
                                    }
                                    ArithBinOp::Sub => {
                                        kernel_body += &format!(
                                        "{TABS}{TMP_PREFIX}{store_target} = {TMP_PREFIX}{lhs} - {TMP_PREFIX}{rhs};\n"
                                    )
                                    }
                                    ArithBinOp::Mul => {
                                        kernel_body += &format!(
                                        "{TABS}{TMP_PREFIX}{store_target} = {TMP_PREFIX}{lhs} * {TMP_PREFIX}{rhs};\n"
                                    )
                                    }
                                    ArithBinOp::Div => {
                                        unreachable!("division should be handled in batched invert")
                                    }
                                },
                                BinOp::Ss(op) => match op {
                                    ArithBinOp::Add => {
                                        kernel_body += &format!(
                                        "{TABS}{TMP_PREFIX}{store_target} = {TMP_PREFIX}{lhs} + {TMP_PREFIX}{rhs};\n"
                                    )
                                    }
                                    ArithBinOp::Sub => {
                                        kernel_body += &format!(
                                        "{TABS}{TMP_PREFIX}{store_target} = {TMP_PREFIX}{lhs} - {TMP_PREFIX}{rhs};\n"
                                    )
                                    }
                                    ArithBinOp::Mul => {
                                        kernel_body += &format!(
                                        "{TABS}{TMP_PREFIX}{store_target} = {TMP_PREFIX}{lhs} * {TMP_PREFIX}{rhs};\n"
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
                                                format!("{TABS}{TMP_PREFIX}{store_target} = (idx == 0 && is_first) ? {TMP_PREFIX}{lhs} + {TMP_PREFIX}{rhs} : {TMP_PREFIX}{rhs};\n")
                                            }
                                            PolyType::Lagrange => {
                                                format!("{TABS}{TMP_PREFIX}{store_target} = {TMP_PREFIX}{lhs} + {TMP_PREFIX}{rhs};\n")
                                            }
                                        };
                                        kernel_body += &stmt;
                                    }
                                    SpOp::Sub => {
                                        let stmt = match self.graph.poly_repr {
                                            PolyType::Coef => {
                                                format!("{TABS}{TMP_PREFIX}{store_target} = (idx == 0 && is_first) ? {TMP_PREFIX}{lhs} - {TMP_PREFIX}{rhs} : {TMP_PREFIX}{rhs}.neg();\n")
                                            }
                                            PolyType::Lagrange => {
                                                format!("{TABS}{TMP_PREFIX}{store_target} = {TMP_PREFIX}{lhs} - {TMP_PREFIX}{rhs};\n")
                                            }
                                        };
                                        kernel_body += &stmt;
                                    }
                                    SpOp::SubBy => {
                                        let stmt = match self.graph.poly_repr {
                                            PolyType::Coef => {
                                                format!("{TABS}{TMP_PREFIX}{store_target} = (idx == 0 && is_first) ? {TMP_PREFIX}{rhs} - {TMP_PREFIX}{lhs} : {TMP_PREFIX}{rhs};\n")
                                            }
                                            PolyType::Lagrange => {
                                                format!("{TABS}{TMP_PREFIX}{store_target} = {TMP_PREFIX}{rhs} - {TMP_PREFIX}{lhs};\n")
                                            }
                                        };
                                        kernel_body += &stmt;
                                    }
                                    SpOp::Mul => {
                                        kernel_body += &format!(
                                        "{TABS}{TMP_PREFIX}{store_target} = {TMP_PREFIX}{lhs} * {TMP_PREFIX}{rhs};\n"
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
                            let arg = arith2reg[arg];
                            let store_target = arith2reg[&cur];
                            match op {
                                UnrOp::P(ArithUnrOp::Neg) => {
                                    kernel_body +=
                                        &format!("{TABS}{TMP_PREFIX}{store_target} = {TMP_PREFIX}{arg}.neg();\n");
                                }
                                UnrOp::P(ArithUnrOp::Inv) => {
                                    unreachable!("invert poly should be handled in batche invert")
                                }
                                UnrOp::P(ArithUnrOp::Pow(power)) => {
                                    kernel_body += &format!(
                                        "{TABS}{TMP_PREFIX}{store_target} = {TMP_PREFIX}{arg}.pow({power});\n"
                                    );
                                }
                                UnrOp::S(ArithUnrOp::Neg) => {
                                    kernel_body +=
                                        &format!("{TABS}{TMP_PREFIX}{store_target} = {TMP_PREFIX}{arg}.neg();\n");
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
            kernel_bodies.push(kernel_body);
        }

        let used_regs = allocator.get_max_used_pages();
        let spill_regs = spill_manager.max_spill_pos;
        let mut kernels = Vec::new();
        for id in 0..(self.partition.len() - 1) {
            let mut kernel = String::new();
            kernel.push_str(&format!("#include \"{}{HEADER_SUFFIX}\"\n", self.name));
            kernel.push_str("namespace detail {\n");

            // generate the kernel signature
            kernel.push_str(&format!(
                "template <typename Field>\n__device__ void {}_{SUB_FUNC_NAME}{id}",
                self.name
            ));
            kernel.push_str(&self.gen_device_func_args("Field", used_regs));
            kernel.push_str(" {\n");

            kernel.push_str(&kernel_bodies[id]);

            kernel.push_str("}\n");

            // explicitly instantiate the template
            kernel += &format!(
                "template __device__ void {}_{SUB_FUNC_NAME}{id}<{FIELD_NAME}>",
                self.name
            );
            kernel.push_str(&self.gen_device_func_args(FIELD_NAME, used_regs));
            kernel.push_str(";\n");

            kernel.push_str("}\n");
            kernels.push(kernel);
        }

        (kernels, used_regs, spill_regs)
    }
}
