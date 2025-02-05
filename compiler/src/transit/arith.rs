use zkpoly_common::{define_usize_id, heap::{Heap, UsizeId}};

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
pub enum Arith<I> {
    Bin(BinOp, I, I),
    Unr(UnrOp, I),
}

#[derive(Debug, Clone)]
pub enum Vertex<I, Ai> {
    A(Arith<Ai>),
    /// Input from computation graph. `.1` is rotation of the input polynomial.
    Input(I, i32),
}

impl<I, Ai> Vertex<I, Ai> {
    pub fn unwrap_input(&self) -> &I {
        match self {
            Vertex::Input(i, _) => i,
            _ => panic!("unwrap_input: not an input"),
        }
    }

    pub fn unwrap_input_mut(&mut self) -> &mut I {
        match self {
            Vertex::Input(i, _) => i,
            _ => panic!("unwrap_input_mut: not an input"),
        }
    }
}

pub mod template {
    use super::*;

    #[derive(Debug, Clone)]
    pub struct Ag<I, Ai> {
        pub(super) output: Vec<Ai>,
        pub(super) inputs: Vec<Ai>,
        pub(super) heap: Heap<Ai, Vertex<I, Ai>>
    }

}

pub type Ag<I> = template::Ag<I, ExprId>;

impl<I> Ag<I> where I: UsizeId {
    pub fn uses_mut<'a>(&'a mut self) -> impl Iterator<Item = &mut I> + 'a {
        self.inputs.iter().map(|&i| self.heap.get_mut(i).unwrap_input_mut())
    }

    pub fn uses(&self) -> impl Iterator<Item = I> {
        self.inputs.iter().map(|&i| self.heap.get(i).unwrap_input().clone())
    }

    pub fn relabeled<I2>(&self, mut mapping: impl FnMut(I) -> I2) -> Ag<I2> {
        let heap = self.heap.map(&mut |_, v| match v {
            Vertex::Input(i, rot) => Vertex::Input(mapping(i), rot),
            Vertex::A(arith) => Vertex::A(arith),
        });

        Ag {
            output: self.output.clone(),
            inputs: self.inputs.clone(),
            heap,
        }
    }
}
