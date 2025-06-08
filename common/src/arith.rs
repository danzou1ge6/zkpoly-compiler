use std::collections::BTreeSet;

use crate::{
    define_usize_id,
    digraph::internal::{Digraph, Predecessors},
    heap::UsizeId,
    typ::PolyType,
};

/// Scalar-Polynomial operator
#[derive(
    Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord, serde::Serialize, serde::Deserialize,
)]
pub enum SpOp {
    Add,
    Sub,
    SubBy,
    Mul,
    Div,
    DivBy,
}

impl SpOp {
    pub fn for_4ops(op: ArithBinOp, rev: bool) -> Self {
        match op {
            ArithBinOp::Add => SpOp::Add,
            ArithBinOp::Sub => {
                if rev {
                    SpOp::SubBy
                } else {
                    SpOp::Sub
                }
            }
            ArithBinOp::Mul => SpOp::Mul,
            ArithBinOp::Div => {
                if rev {
                    SpOp::DivBy
                } else {
                    SpOp::Div
                }
            }
        }
    }

    pub fn support_coef(&self) -> bool {
        match self {
            Self::Div => false,
            _ => true,
        }
    }
}

mod op_template {
    /// Binary operator.
    /// [`P`]: polynomial-Polynomial opertor
    #[derive(
        Debug, Clone, Hash, PartialEq, Eq, PartialOrd, Ord, serde::Serialize, serde::Deserialize,
    )]
    pub enum BinOp<Pp, Ss, Sp> {
        Pp(Pp),
        Ss(Ss),
        Sp(Sp),
    }

    /// Unary operator.
    /// [`Po`]: polynomial unary operator
    #[derive(
        Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord, serde::Deserialize, serde::Serialize,
    )]
    pub enum UnrOp<Po, So> {
        P(Po),
        S(So),
    }
}

#[derive(
    Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord, serde::Serialize, serde::Deserialize,
)]
pub enum ArithBinOp {
    Add,
    Sub,
    Mul,
    Div,
}

impl ArithBinOp {
    pub fn support_coef(&self) -> bool {
        match self {
            ArithBinOp::Add | ArithBinOp::Sub => true,
            _ => false,
        }
    }
}

#[derive(
    Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord, serde::Serialize, serde::Deserialize,
)]
pub enum ArithUnrOp {
    Neg,
    Inv,
    Pow(u64),
}

impl ArithUnrOp {
    pub fn support_coef(&self) -> bool {
        match self {
            ArithUnrOp::Neg => true,
            _ => false,
        }
    }
}

pub type BinOp = op_template::BinOp<ArithBinOp, ArithBinOp, SpOp>;
pub type UnrOp = op_template::UnrOp<ArithUnrOp, ArithUnrOp>;

define_usize_id!(ExprId);

/// Kind-specific data of expressions.
#[derive(
    Debug, Clone, Hash, PartialEq, Eq, PartialOrd, Ord, serde::Serialize, serde::Deserialize,
)]
pub enum Arith<Index> {
    Bin(BinOp, Index, Index),
    Unr(UnrOp, Index),
}

impl<I> Arith<I>
where
    I: Clone,
{
    pub fn uses_ref<'s>(&'s self) -> Box<dyn Iterator<Item = &'s I> + 's> {
        match self {
            Arith::Bin(_, lhs, rhs) => Box::new([lhs, rhs].into_iter()),
            Arith::Unr(_, rhs) => Box::new([rhs].into_iter()),
        }
    }

    pub fn uses<'s>(&'s self) -> impl Iterator<Item = I> + 's {
        self.uses_ref().cloned()
    }

    pub fn uses_mut<'a>(&'a mut self) -> Box<dyn Iterator<Item = &'a mut I> + 'a> {
        match self {
            Arith::Bin(_, lhs, rhs) => Box::new([lhs, rhs].into_iter()),
            Arith::Unr(_, rhs) => Box::new([rhs].into_iter()),
        }
    }

    pub fn relabeled<I2>(&self, f: &mut impl FnMut(I) -> I2) -> Arith<I2> {
        match self {
            Arith::Bin(op, lhs, rhs) => Arith::Bin(op.clone(), f(lhs.clone()), f(rhs.clone())),
            Arith::Unr(op, rhs) => Arith::Unr(op.clone(), f(rhs.clone())),
        }
    }

    pub fn try_relabeled<I2, Er>(
        &self,
        f: &mut impl FnMut(I) -> Result<I2, Er>,
    ) -> Result<Arith<I2>, Er> {
        let r = match self {
            Arith::Bin(op, lhs, rhs) => Arith::Bin(op.clone(), f(lhs.clone())?, f(rhs.clone())?),
            Arith::Unr(op, rhs) => Arith::Unr(op.clone(), f(rhs.clone())?),
        };

        Ok(r)
    }
}

#[derive(
    Debug, Clone, Copy, PartialEq, Hash, Eq, PartialOrd, Ord, serde::Serialize, serde::Deserialize,
)]
pub enum FusedType {
    Scalar,
    ScalarArray,
}

#[derive(
    Debug, Clone, PartialEq, Hash, Eq, PartialOrd, Ord, serde::Serialize, serde::Deserialize,
)]
pub enum Operation<OuterId, ArithIndex> {
    Arith(Arith<ArithIndex>),
    Input {
        outer_id: OuterId,
        typ: FusedType,
        mutability: Mutability,
    },
    Output {
        outer_id: OuterId,
        typ: FusedType,
        store_node: ArithIndex,
        in_node: Option<ArithIndex>,
    },
    Todo,
    // 0 is the output index, 1 is the index of the data to be stored
    // 2: the output's outer index is the same as some inputs' outer index
}

#[derive(
    Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord, serde::Serialize, serde::Deserialize,
)]
pub struct ArithVertex<OuterId, ArithIndex> {
    pub op: Operation<OuterId, ArithIndex>,
}

impl<OuterId, ArithIndex> Operation<OuterId, ArithIndex> {
    pub fn unwrap_temp(&self) -> &Arith<ArithIndex> {
        match self {
            Self::Arith(arith) => arith,
            _ => panic!("Vertex is not an arithmetic expression"),
        }
    }
    pub fn unwrap_input_outerid(&self) -> &OuterId {
        match self {
            Self::Input { outer_id, .. } => outer_id,
            _ => panic!("Vertex is not an input"),
        }
    }

    pub fn unwrap_input_typ(&self) -> FusedType {
        match self {
            Self::Input { typ, .. } => typ.clone(),
            _ => panic!("Vertex is not an input"),
        }
    }

    pub fn unwrap_input_mutability(&self) -> Mutability {
        match self {
            Self::Input { mutability, .. } => mutability.clone(),
            _ => panic!("Vertex is not an input"),
        }
    }

    pub fn unwrap_input_mut(&mut self) -> (&mut OuterId, &mut FusedType, &mut Mutability) {
        match self {
            Self::Input {
                outer_id,
                typ,
                mutability,
            } => (outer_id, typ, mutability),
            _ => panic!("Vertex is not an input"),
        }
    }

    pub fn unwrap_output(&self) -> (&OuterId, &FusedType, &ArithIndex, &Option<ArithIndex>) {
        match self {
            Operation::Output {
                outer_id,
                typ,
                store_node,
                in_node,
            } => (outer_id, typ, store_node, in_node),
            _ => panic!("Vertex is not an output"),
        }
    }

    pub fn unwrap_output_mut(
        &mut self,
    ) -> (
        &mut OuterId,
        &mut FusedType,
        &mut ArithIndex,
        &mut Option<ArithIndex>,
    ) {
        match self {
            Operation::Output {
                outer_id,
                typ,
                store_node,
                in_node,
            } => (outer_id, typ, store_node, in_node),
            _ => panic!("Vertex is not an output"),
        }
    }
}

impl<OuterId, ArithIndex> ArithVertex<OuterId, ArithIndex>
where
    ArithIndex: Copy + 'static,
{
    pub fn uses<'s>(&'s self) -> Box<dyn Iterator<Item = ArithIndex> + 's> {
        match &self.op {
            Operation::Arith(a) => Box::new(a.uses()),
            Operation::Output {
                store_node,
                in_node,
                ..
            } => {
                let src = in_node.clone();
                Box::new(src.into_iter().chain(std::iter::once(*store_node)))
            }
            _ => Box::new(std::iter::empty()),
        }
    }

    pub fn uses_mut<'s>(&'s mut self) -> Box<dyn Iterator<Item = &'s mut ArithIndex> + 's> {
        match &mut self.op {
            Operation::Arith(a) => a.uses_mut(),
            Operation::Output {
                store_node,
                in_node,
                ..
            } => {
                let store = store_node as *mut ArithIndex;
                let mut result = Vec::new();
                result.extend(in_node.iter_mut());
                // SAFETY: store_node is a unique mutable reference
                unsafe { result.push(&mut *store) };
                Box::new(result.into_iter())
            }
            _ => Box::new(std::iter::empty()),
        }
    }
}

#[derive(
    Debug, Clone, Copy, PartialEq, Hash, Eq, PartialOrd, Ord, serde::Serialize, serde::Deserialize,
)]
pub enum Mutability {
    Const,
    Mut,
}

// DAG for arithmetic expressions.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, serde::Serialize, serde::Deserialize)]
pub struct ArithGraph<OuterId, ArithIndex> {
    pub outputs: Vec<ArithIndex>, // output ids
    pub inputs: Vec<ArithIndex>,  // input ids
    pub g: Digraph<ArithIndex, ArithVertex<OuterId, ArithIndex>>,
    pub poly_repr: PolyType,
    pub poly_degree: Option<usize>,
}

impl<OuterId, ArithIndex> ArithGraph<OuterId, ArithIndex>
where
    ArithIndex: UsizeId,
    OuterId: Clone,
{
    pub fn decide_chunking<T: Sized>(&mut self, usable_size: u64) -> Option<u64> {
        let (inputs_space, outputs_space) = self.space_needed::<T>();

        if ((inputs_space + outputs_space) as f64) < usable_size as f64 * 0.8 {
            None
        } else {
            let mut chunking = 4;
            let total_space = 2 * inputs_space + 3 * outputs_space;

            // helper func
            let div_ceil = |a: usize, b: u64| (a as u64 + b - 1) / b;

            while div_ceil(total_space, chunking) * 3 > usable_size {
                chunking *= 2;
            }
            assert!(self.poly_degree.unwrap() as u64 % chunking == 0);
            Some(chunking)
        }
    }

    pub fn reorder_input_outputs(&mut self) {
        // reorder into scalars  polys order
        let mut inputs = Vec::new();
        let mut outputs = Vec::new();

        // first push all scalars
        for id in self.inputs.iter() {
            let v = self.g.vertex(*id);
            if let Operation::Input { typ, .. } = &v.op {
                if typ == &FusedType::Scalar {
                    inputs.push(*id);
                }
            } else {
                panic!("arith vertex in the inputs table should be inputs")
            }
        }
        for id in self.outputs.iter() {
            let v = self.g.vertex(*id);
            if let Operation::Output { typ, .. } = &v.op {
                if typ == &FusedType::Scalar {
                    outputs.push(*id);
                }
            } else {
                panic!("arith vertex in the outputs table should be outputs")
            }
        }

        // then push all polys
        for id in self.inputs.iter() {
            let v = self.g.vertex(*id);
            if let Operation::Input { typ, .. } = &v.op {
                if typ == &FusedType::ScalarArray {
                    inputs.push(*id);
                }
            } else {
                panic!("arith vertex in the inputs table should be inputs")
            }
        }
        for id in self.outputs.iter() {
            let v = self.g.vertex(*id);
            if let Operation::Output { typ, .. } = &v.op {
                if typ == &FusedType::ScalarArray {
                    outputs.push(*id);
                }
            } else {
                panic!("arith vertex in the outputs table should be outputs")
            }
        }

        self.inputs = inputs;
        self.outputs = outputs;
    }

    pub fn get_inplace_inputs(&self) -> BTreeSet<ArithIndex> {
        // collect the inputs that are also outputs
        let mut inplace_inputs = BTreeSet::new();
        for inner_id in self.outputs.iter() {
            if let Operation::Output { in_node, .. } = &self.g.vertex(*inner_id).op {
                if let Some(in_id) = in_node {
                    let m = self.g.vertex(*in_id).op.unwrap_input_mutability();
                    if m == Mutability::Mut {
                        inplace_inputs.insert(*in_id);
                    }
                }
            } else {
                unreachable!("output should be an Operation::Output");
            }
        }
        inplace_inputs
    }

    pub fn gen_var_lists(&self) -> (Vec<(FusedType, ArithIndex)>, Vec<(FusedType, ArithIndex)>) {
        let mut vars = Vec::new();
        let mut mut_vars = Vec::new();
        for inner_id in self.outputs.iter() {
            if let Operation::Output { typ, .. } = &self.g.vertex(*inner_id).op {
                mut_vars.push((typ.clone(), *inner_id));
            } else {
                unreachable!("output should be an Operation::Output");
            }
        }

        let inplace_inputs = self.get_inplace_inputs();

        for inner_id in self.inputs.iter() {
            if let Operation::Input { typ, .. } = &self.g.vertex(*inner_id).op {
                // skip the inputs that are also outputs
                if inplace_inputs.contains(inner_id) {
                    continue;
                }
                vars.push((typ.clone(), *inner_id));
            } else {
                unreachable!("input should be an Operation::Input");
            }
        }

        (vars, mut_vars)
    }

    pub fn space_needed<T: Sized>(&self) -> (usize, usize) {
        let poly_space = self.poly_degree.unwrap() * size_of::<T>();
        let scalar_space = size_of::<T>();
        let space_for_ft = |t| match t {
            FusedType::Scalar => scalar_space,
            FusedType::ScalarArray => poly_space,
        };
        let inputs_space = self
            .inputs
            .iter()
            .map(|i| space_for_ft(self.g.vertex(*i).op.unwrap_input_typ()))
            .chain(self.outputs.iter().filter_map(|i| {
                let (_, ft, _, in_node) = self.g.vertex(*i).op.unwrap_output();
                if in_node.is_some() {
                    None
                } else {
                    Some(space_for_ft(*ft))
                }
            }))
            .sum();

        let outputs_space = self
            .outputs
            .iter()
            .map(|i| {
                let (_, ft, _, in_node) = self.g.vertex(*i).op.unwrap_output();
                if in_node.is_some() {
                    0
                } else {
                    space_for_ft(*ft)
                }
            })
            .sum();

        (inputs_space, outputs_space)
    }

    pub fn uses<'s>(&'s self) -> impl Iterator<Item = OuterId> + 's {
        self.inputs
            .iter()
            .map(|&i| self.g.vertex(i).op.unwrap_input_outerid().clone())
    }

    pub fn uses_ref<'s>(&'s self) -> impl Iterator<Item = &'s OuterId> + 's {
        self.inputs
            .iter()
            .map(|&i| self.g.vertex(i).op.unwrap_input_outerid())
    }

    pub fn mutable_uses<'a>(&'a self) -> Box<dyn Iterator<Item = OuterId> + 'a> {
        let mut results = Vec::new();
        self.g.0 .0.iter().for_each(|v| {
            if let Operation::Input {
                outer_id,
                mutability,
                ..
            } = &v.op
            {
                if *mutability == Mutability::Mut {
                    results.push(outer_id.clone());
                }
            }
        });
        Box::new(results.into_iter())
    }

    pub fn mutable_uses_mut<'a>(&'a mut self) -> Box<dyn Iterator<Item = &'a mut OuterId> + 'a> {
        let mut results = Vec::new();
        self.g.0 .0.iter_mut().for_each(|v| {
            if let Operation::Input {
                outer_id,
                mutability,
                ..
            } = &mut v.op
            {
                if *mutability == Mutability::Mut {
                    results.push(outer_id);
                }
            }
        });
        Box::new(results.into_iter())
    }

    pub fn outputs_inplace<'a, 'b>(&'b self) -> Box<dyn Iterator<Item = Option<OuterId>> + 'b> {
        let results = self
            .outputs
            .iter()
            .map(|output_id| {
                if let Operation::Output { in_node, .. } = &self.g.vertex(*output_id).op {
                    in_node.map(|in_id| self.g.vertex(in_id).op.unwrap_input_outerid().clone())
                } else {
                    panic!("output nodes should have op Output")
                }
            })
            .collect::<Vec<_>>();

        Box::new(results.into_iter())
    }

    pub fn uses_mut<'s>(&'s mut self) -> Box<dyn Iterator<Item = &'s mut OuterId> + 's> {
        let mut results = Vec::new();
        self.g.0 .0.iter_mut().for_each(|v| {
            if let Operation::Input { outer_id, .. } = &mut v.op {
                results.push(outer_id);
            }
        });
        Box::new(results.into_iter())
    }

    pub fn try_relabeled<I2: Default + Ord + std::fmt::Debug, Er>(
        &self,
        mut mapping: impl FnMut(OuterId) -> Result<I2, Er>,
    ) -> Result<ArithGraph<I2, ArithIndex>, Er> {
        let mut used_outer_ids = BTreeSet::new(); // as we have done CSE, outer ids should be unique
        let heap = self
            .g
            .map_by_ref_result(&mut |_, v: &ArithVertex<OuterId, ArithIndex>| {
                let op = match &v.op {
                    Operation::Input {
                        outer_id,
                        typ,
                        mutability,
                    } => {
                        let new_outer_id = mapping(outer_id.clone())?;
                        if used_outer_ids.contains(&new_outer_id) {
                            panic!("duplicated outer id {:?}", new_outer_id)
                        }
                        used_outer_ids.insert(new_outer_id);
                        Operation::Input {
                            outer_id: mapping(outer_id.clone())?,
                            typ: typ.clone(),
                            mutability: mutability.clone(),
                        }
                    }
                    Operation::Arith(arith) => Operation::Arith(arith.clone()),
                    Operation::Output {
                        typ,
                        store_node,
                        in_node,
                        ..
                    } => Operation::Output {
                        // This field is meaningless before kernel generation
                        outer_id: I2::default(),
                        typ: typ.clone(),
                        store_node: *store_node,
                        in_node: in_node.clone(),
                    },
                    Operation::Todo => Operation::Todo,
                };

                Ok(ArithVertex { op })
            })?;

        Ok(ArithGraph {
            outputs: self.outputs.clone(),
            inputs: self.inputs.clone(),
            g: heap,
            poly_repr: self.poly_repr.clone(),
            poly_degree: self.poly_degree,
        })
    }

    pub fn relabeled<I2: Default + Ord + std::fmt::Debug>(
        &self,
        mut mapping: impl FnMut(OuterId) -> I2,
    ) -> ArithGraph<I2, ArithIndex> {
        self.try_relabeled::<_, ()>(|outer_id| Ok(mapping(outer_id)))
            .unwrap()
    }
}

impl<OuterId, ArithIndex> Predecessors<ArithIndex> for ArithVertex<OuterId, ArithIndex>
where
    ArithIndex: Copy + 'static,
{
    fn predecessors(&self) -> impl Iterator<Item = ArithIndex> {
        self.uses()
    }
}

impl<OuterId, ArithIndex> ArithGraph<OuterId, ArithIndex>
where
    ArithIndex: UsizeId + 'static,
{
    pub fn topology_sort<'s>(
        &'s self,
    ) -> impl Iterator<Item = (ArithIndex, &'s ArithVertex<OuterId, ArithIndex>)> + 's {
        self.g.topology_sort()
    }
}

pub fn check_degree_of_todo_vertices<
    OuterId: UsizeId,
    InnerId: UsizeId + std::fmt::Debug + 'static,
>(
    name: String,
    ag: &ArithGraph<OuterId, InnerId>,
) {
    let out_deg = ag.g.degrees_out();
    for vid in ag.g.vertices() {
        let v = ag.g.vertex(vid);
        match &v.op {
            Operation::Todo => {
                if out_deg[vid] != 0 {
                    panic!(
                        "Todo node {:?} in graph {} has non-zero out-degree {}",
                        vid, name, out_deg[vid]
                    );
                }
            }
            _ => {}
        }
    }
}

pub mod hash;
pub mod partition;
pub mod pretty_print;
pub mod scheduler;
pub mod visualize;
