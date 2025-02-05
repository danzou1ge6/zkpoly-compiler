use crate::{
    define_usize_id,
    heap::{Heap, UsizeId},
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
    Eval,
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

#[derive(Debug, Clone)]
pub enum Operation<VarType, ArithIndex> {
    Arith(Arith<ArithIndex>),
    Input(VarType),
    Output(VarType, ArithIndex),
}

#[derive(Debug, Clone)]
pub struct Vertex<VarType, ArithIndex> {
    pub op: Operation<VarType, ArithIndex>,
    pub target: Vec<ArithIndex>,
}

impl<VarType, ArithIndex> Operation<VarType, ArithIndex> {
    pub fn unwrap_temp(&self) -> &Arith<ArithIndex> {
        match self {
            Self::Arith(arith) => arith,
            _ => panic!("Vertex is not an arithmetic expression"),
        }
    }
    pub fn unwrap_global(&self) -> &VarType {
        match self {
            Self::Input(var) => var,
            _ => panic!("Vertex is not an input"),
        }
    }
}

// DAG for arithmetic expressions.
#[derive(Debug, Clone)]
pub struct ArithGraph<VarType, ArithIndex> {
    pub outputs: Vec<ArithIndex>, // output ids
    pub inputs: Vec<ArithIndex>,  // input ids
    pub heap: Heap<ArithIndex, Vertex<VarType, ArithIndex>>,
}

impl<VarType, ArithIndex> ArithGraph<VarType, ArithIndex>
where
    ArithIndex: UsizeId,
    VarType: Copy,
{
    pub fn uses<'s>(&'s self) -> impl Iterator<Item = VarType> + 's {
        self.inputs
            .iter()
            .map(|&i| self.heap[i].op.unwrap_global().clone())
    }

    pub fn relabeled<I2>(
        &self,
        mut mapping: impl FnMut(VarType) -> I2,
    ) -> ArithGraph<I2, ArithIndex> {
        let heap = self.heap.map_by_ref(&mut |_, v| {
            let op = match &v.op {
                Operation::Input(i) => Operation::Input(mapping(*i)),
                Operation::Arith(arith) => Operation::Arith(arith.clone()),
                Operation::Output(i, o) => Operation::Output(mapping(*i), *o),
            };

            Vertex {
                op,
                target: v.target.clone(),
            }
        });

        ArithGraph {
            outputs: self.outputs.clone(),
            inputs: self.inputs.clone(),
            heap,
        }
    }
}
