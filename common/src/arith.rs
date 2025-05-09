use std::collections::BTreeSet;

use crate::{
    define_usize_id,
    digraph::internal::{Digraph, Predecessors},
    heap::{Heap, UsizeId},
    typ::PolyType,
};

/// Scalar-Polynomial operator
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
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
    #[derive(Debug, Clone, Hash, PartialEq, Eq, PartialOrd, Ord)]
    pub enum BinOp<Pp, Ss, Sp> {
        Pp(Pp),
        Ss(Ss),
        Sp(Sp),
    }

    /// Unary operator.
    /// [`Po`]: polynomial unary operator
    #[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
    pub enum UnrOp<Po, So> {
        P(Po),
        S(So),
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
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

#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
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
#[derive(Debug, Clone, Hash, PartialEq, Eq, PartialOrd, Ord)]
pub enum Arith<Index> {
    Bin(BinOp, Index, Index),
    Unr(UnrOp, Index),
}

impl<I> Arith<I>
where
    I: Copy,
{
    pub fn uses<'s>(&'s self) -> Box<dyn Iterator<Item = I> + 's> {
        match self {
            Arith::Bin(_, lhs, rhs) => Box::new([*lhs, *rhs].into_iter()),
            Arith::Unr(_, rhs) => Box::new([*rhs].into_iter()),
        }
    }

    pub fn uses_mut<'a>(&'a mut self) -> Box<dyn Iterator<Item = &'a mut I> + 'a> {
        match self {
            Arith::Bin(_, lhs, rhs) => Box::new([lhs, rhs].into_iter()),
            Arith::Unr(_, rhs) => Box::new([rhs].into_iter()),
        }
    }

    pub fn relabeled<I2>(&self, f: &mut impl FnMut(I) -> I2) -> Arith<I2> {
        match self {
            Arith::Bin(op, lhs, rhs) => Arith::Bin(op.clone(), f(*lhs), f(*rhs)),
            Arith::Unr(op, rhs) => Arith::Unr(op.clone(), f(*rhs)),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Hash, Eq, PartialOrd, Ord)]
pub enum FusedType {
    Scalar,
    ScalarArray,
}

#[derive(Debug, Clone, PartialEq, Hash, Eq, PartialOrd, Ord)]
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
        in_node: Vec<ArithIndex>,
    },
    Todo,
    // 0 is the output index, 1 is the index of the data to be stored
    // 2: the output's outer index is the same as some inputs' outer index
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
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

    pub fn unwrap_output(&self) -> (&OuterId, &FusedType, &ArithIndex, &[ArithIndex]) {
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
        &mut Vec<ArithIndex>,
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
            Operation::Arith(a) => a.uses(),
            Operation::Output {
                store_node,
                in_node,
                ..
            } => {
                let mut src = in_node.clone();
                src.push(*store_node);
                Box::new(src.into_iter())
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

#[derive(Debug, Clone, Copy, PartialEq, Hash, Eq, PartialOrd, Ord)]
pub enum Mutability {
    Const,
    Mut,
}

// DAG for arithmetic expressions.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct ArithGraph<OuterId, ArithIndex> {
    pub outputs: Vec<ArithIndex>, // output ids
    pub inputs: Vec<ArithIndex>,  // input ids
    pub g: Digraph<ArithIndex, ArithVertex<OuterId, ArithIndex>>,
    pub poly_repr: PolyType,
    pub poly_degree: Option<usize>,
}

impl<OuterId, ArithIndex> ArithGraph<OuterId, ArithIndex>
where
    ArithIndex: UsizeId + 'static,
    OuterId: UsizeId,
{
    pub fn change_mutability(
        &mut self,
        succ: &Heap<OuterId, BTreeSet<OuterId>>,
        poly_limit: usize,
    ) -> Vec<ArithIndex> {
        // change mutability of poly, scalars must be immutable
        let succ_arith = self.g.successors();
        let mut mutable_inputs = Vec::new();
        for id in self.inputs.iter() {
            let v = self.g.vertex_mut(*id);
            if let Operation::Input {
                outer_id,
                typ,
                mutability,
            } = &mut v.op
            {
                if succ_arith[*id].len() == succ[*outer_id].len() {
                    if *typ == FusedType::ScalarArray && mutable_inputs.len() < poly_limit {
                        mutable_inputs.push(*id);
                        *mutability = Mutability::Mut;
                    }
                }
            } else {
                panic!("arith vertex in the inputs table should be inputs")
            }
        }
        mutable_inputs
    }
}

impl<OuterId, ArithIndex> ArithGraph<OuterId, ArithIndex>
where
    ArithIndex: UsizeId,
    OuterId: Copy,
{
    pub fn get_inplace_inputs(&self) -> BTreeSet<ArithIndex> {
        // collect the inputs that are also outputs
        let mut inplace_inputs = BTreeSet::new();
        for inner_id in self.outputs.iter() {
            if let Operation::Output { in_node, .. } = &self.g.vertex(*inner_id).op {
                for in_id in in_node {
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

    pub fn space_needed<T: Sized>(&self) -> usize {
        let poly_space = self.poly_degree.unwrap() * size_of::<T>();
        let scalar_space = size_of::<T>();
        let space_for_ft = |t| match t {
            FusedType::Scalar => scalar_space,
            FusedType::ScalarArray => poly_space,
        };
        self.inputs
            .iter()
            .map(|i| space_for_ft(self.g.vertex(*i).op.unwrap_input_typ()))
            .chain(self.outputs.iter().filter_map(|i| {
                let (_, ft, _, in_node) = self.g.vertex(*i).op.unwrap_output();
                if !in_node.is_empty() {
                    None
                } else {
                    Some(space_for_ft(*ft))
                }
            }))
            .sum()
    }

    pub fn uses<'s>(&'s self) -> impl Iterator<Item = OuterId> + 's {
        self.inputs
            .iter()
            .map(|&i| self.g.vertex(i).op.unwrap_input_outerid().clone())
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
                    results.push(*outer_id);
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
                    if in_node.len() > 0 {
                        Some(*self.g.vertex(in_node[0].clone()).op.unwrap_input_outerid())
                    } else {
                        None
                    }
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
    pub fn relabeled<I2: Default + Ord + std::fmt::Debug>(
        &self,
        mut mapping: impl FnMut(OuterId) -> I2,
    ) -> ArithGraph<I2, ArithIndex> {
        let mut used_outer_ids = BTreeSet::new(); // as we have done CSE, outer ids should be unique
        let heap = self.g.map_by_ref(&mut |_, v| {
            let op = match &v.op {
                Operation::Input {
                    outer_id,
                    typ,
                    mutability,
                } => {
                    let new_outer_id = mapping(*outer_id);
                    if used_outer_ids.contains(&new_outer_id) {
                        panic!("duplicated outer id {:?}", new_outer_id)
                    }
                    used_outer_ids.insert(new_outer_id);
                    Operation::Input {
                        outer_id: mapping(*outer_id),
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

            ArithVertex { op }
        });

        ArithGraph {
            outputs: self.outputs.clone(),
            inputs: self.inputs.clone(),
            g: heap,
            poly_repr: self.poly_repr.clone(),
            poly_degree: self.poly_degree,
        }
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
pub mod pretty_print;
pub mod visualize;
