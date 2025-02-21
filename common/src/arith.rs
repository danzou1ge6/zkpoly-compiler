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
            Self::Div=> false,
            _ => true,
        }
    }
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

impl ArithBinOp {
    pub fn support_coef(&self) -> bool {
        match self {
            ArithBinOp::Add | ArithBinOp::Sub => true,
            _ => false,
        }
    }
}

#[derive(Debug, Clone)]
pub enum ArithUnrOp {
    Neg,
    Inv,
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
#[derive(Debug, Clone)]
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
    pub fn unwrap_global(&self) -> &OuterId {
        match self {
            Self::Input { outer_id, .. } => outer_id,
            _ => panic!("Vertex is not an input"),
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
                let mut src = Vec::new();
                in_node.iter().for_each(|&i| src.push(i));
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
                let mut result = Vec::with_capacity(in_node.len() + 1);
                result.extend(in_node.iter_mut());
                // SAFETY: store_node is a unique mutable reference
                unsafe { result.push(&mut *store) };
                Box::new(result.into_iter())
            }
            _ => Box::new(std::iter::empty()),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum Mutability {
    Const,
    Mut,
}

// DAG for arithmetic expressions.
#[derive(Debug, Clone)]
pub struct ArithGraph<OuterId, ArithIndex> {
    pub outputs: Vec<ArithIndex>, // output ids
    pub inputs: Vec<ArithIndex>,  // input ids
    pub g: Digraph<ArithIndex, ArithVertex<OuterId, ArithIndex>>,
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

    pub fn outputs_inplace<'a, 'b>(
        &'b self,
        mut mutable_scalars: usize,
        mut mutable_polys: usize,
    ) -> Box<dyn Iterator<Item = Option<OuterId>> + 'b> {
        let mut results = Vec::new();
        self.g.0 .0.iter().for_each(|v| {
            if let Operation::Input {
                outer_id,
                mutability,
                typ,
            } = &v.op
            {
                match typ {
                    FusedType::Scalar => {
                        if mutable_scalars > 0 && *mutability == Mutability::Mut {
                            mutable_scalars -= 1;
                            results.push(Some(*outer_id));
                        } else {
                            results.push(None);
                        }
                    }
                    FusedType::ScalarArray => {
                        if mutable_polys > 0 && *mutability == Mutability::Mut {
                            mutable_polys -= 1;
                            results.push(Some(*outer_id));
                        } else {
                            results.push(None);
                        }
                    }
                }
            }
        });
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

            ArithVertex { op }
        });

        ArithGraph {
            outputs: self.outputs.clone(),
            inputs: self.inputs.clone(),
            g: heap,
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
