use crate::{define_usize_id, heap::Heap};

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
