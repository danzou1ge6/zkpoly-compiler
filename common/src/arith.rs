use crate::{
    define_usize_id,
    digraph::internal::{Digraph, Predecessors},
    heap::UsizeId,
};

/// Scalar-Polynomial operator
#[derive(Debug, Clone)]
pub enum SpOp {
    Add,
    Sub,
    SubBy,
    Mul,
    Div,
    DivBy,
}

/// Unary polynomial operator
#[derive(Debug, Clone)]
pub enum POp {
    Neg,
    Inv,
}

mod op_template {
    /// Binary operator.
    /// [`P`]: polynomial-Polynomial opertor
    #[derive(Debug, Clone)]
    pub enum BinOp<Pp, Ss, Sp> {
        Pp(Pp),
        Ss(Ss),
        Sp(Sp),
    }

    /// Unary operator.
    /// [`Po`]: polynomial unary operator
    #[derive(Debug, Clone)]
    pub enum UnrOp<Po, So> {
        P(Po),
        S(So),
    }
}

#[derive(Debug, Clone)]
pub enum ArithBinOp {
    Add,
    Sub,
    Mul,
    Div,
}

#[derive(Debug, Clone)]
pub enum ArithUnrOp {
    Neg,
    Inv,
}

pub type BinOp = op_template::BinOp<ArithBinOp, ArithBinOp, SpOp>;
pub type UnrOp = op_template::UnrOp<POp, ArithUnrOp>;

define_usize_id!(ExprId);

/// Kind-specific data of expressions.
#[derive(Debug, Clone)]
pub enum Arith<Index> {
    Bin(BinOp, Index, Index),
    Unr(UnrOp, Index),
}

#[derive(Debug, Clone, PartialEq)]
pub enum FusedType {
    Scalar,
    ScalarArray,
}

#[derive(Debug, Clone)]
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
    // 0 is the output index, 1 is the index of the data to be stored
    // 2: the output's outer index is the same as some inputs' outer index
}

#[derive(Debug, Clone)]
pub struct Vertex<OuterId, ArithIndex> {
    pub op: Operation<OuterId, ArithIndex>,
    pub target: Vec<ArithIndex>,
}

impl<OuterId, ArithIndex> Operation<OuterId, ArithIndex> {
    pub fn unwrap_temp(&self) -> &Arith<ArithIndex> {
        match self {
            Self::Arith(arith) => arith,
            _ => panic!("Vertex is not an arithmetic expression"),
        }
    }
    pub fn unwrap_global(&self) -> &OuterId {
        match self {
            Self::Input { outer_id, .. } => outer_id,
            _ => panic!("Vertex is not an input"),
        }
    }
}

impl<OuterId, ArithIndex> Vertex<OuterId, ArithIndex>
where
    ArithIndex: Copy + 'static,
{
    pub fn uses(&self) -> Box<dyn Iterator<Item = ArithIndex>> {
        match &self.op {
            Operation::Arith(Arith::Bin(_, lhs, rhs)) => Box::new([*lhs, *rhs].into_iter()),
            Operation::Arith(Arith::Unr(_, rhs)) => Box::new([*rhs].into_iter()),
            Operation::Output {
                store_node,
                in_node,
                ..
            } => {
                let mut src = Vec::new();
                in_node.iter().for_each(|&i| src.push(i));
                src.push(*store_node);
                Box::new(src.into_iter())
            }
            _ => Box::new(std::iter::empty()),
        }
    }
}

#[derive(Debug, Clone)]
pub enum Mutability {
    Const,
    Mut,
}

// DAG for arithmetic expressions.
#[derive(Debug, Clone)]
pub struct ArithGraph<OuterId, ArithIndex> {
    pub outputs: Vec<ArithIndex>, // output ids
    pub inputs: Vec<ArithIndex>,  // input ids
    pub g: Digraph<ArithIndex, Vertex<OuterId, ArithIndex>>,
}

impl<OuterId, ArithIndex> ArithGraph<OuterId, ArithIndex>
where
    ArithIndex: UsizeId,
    OuterId: Copy,
{
    pub fn uses<'s>(&'s self) -> impl Iterator<Item = OuterId> + 's {
        self.inputs
            .iter()
            .map(|&i| self.g.vertex(i).op.unwrap_global().clone())
    }

    pub fn relabeled<I2>(
        &self,
        mut mapping: impl FnMut(OuterId) -> I2,
    ) -> ArithGraph<I2, ArithIndex> {
        let heap = self.g.map_by_ref(&mut |_, v| {
            let op = match &v.op {
                Operation::Input {
                    outer_id,
                    typ,
                    mutability,
                } => Operation::Input {
                    outer_id: mapping(*outer_id),
                    typ: typ.clone(),
                    mutability: mutability.clone(),
                },
                Operation::Arith(arith) => Operation::Arith(arith.clone()),
                Operation::Output {
                    outer_id,
                    typ,
                    store_node,
                    in_node,
                } => Operation::Output {
                    outer_id: mapping(*outer_id),
                    typ: typ.clone(),
                    store_node: *store_node,
                    in_node: in_node.clone(),
                },
            };

            Vertex {
                op,
                target: v.target.clone(),
            }
        });

        ArithGraph {
            outputs: self.outputs.clone(),
            inputs: self.inputs.clone(),
            g: heap,
        }
    }
}

impl<OuterId, ArithIndex> Predecessors<ArithIndex> for Vertex<OuterId, ArithIndex>
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
    ) -> impl Iterator<Item = (ArithIndex, &'s Vertex<OuterId, ArithIndex>)> + 's {
        self.g.topology_sort()
    }
}
