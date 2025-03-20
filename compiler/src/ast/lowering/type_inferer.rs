use super::*;

#[derive(Debug, Clone)]
pub enum ErrorNode<Rt: RuntimeType> {
    ExtendToSmallerDegree {
        from: u64,
        to: u64,
    },
    ExpectPolynomial,
    ExpectPolynomialType(PolyType),
    ArithOnDifferentPolynomialTypes,
    ArithOnDifferentDegreeLagrangePolynomials(u64, u64),
    BadSlice {
        begin: u64,
        end: u64,
        deg: u64,
    },
    ExpectScalar,
    InterpolateArgsDifferentLength(usize, usize),
    ArrayOfInconsistentTypes(usize, type2::Typ<Rt>, usize, type2::Typ<Rt>),
    ExpectPointBase,
    MsmInconsistentInputLengths {
        points_len: u64,
        poly_deg: u64,
        poly_i: usize,
    },
    MsmInconsistentInputPolyTypes(usize, PolyType, usize, PolyType),
    UnhashableToTranscript(type2::Typ<Rt>),
    ExpectTuple,
    TupleIndexOutofBound {
        index: usize,
        len: usize,
    },
    ExpectArray,
    ArrayIndexOutofBound {
        index: usize,
        len: usize,
    },
    ExpectTranscript,
    KateDivisionPolyDegreeTooSmall,
    IncompatibleWithAnnotation(type2::Typ<Rt>, Typ<Rt>),
    IndexPolyOutOfBound {
        index: u64,
        len: u64,
    },
    AssertDifferentTypes(type2::Typ<Rt>, type2::Typ<Rt>),
}

#[derive(Debug, Clone)]
pub struct Error<'s, Rt: RuntimeType> {
    pub node: ErrorNode<Rt>,
    pub at: SourceInfo<'s>,
    pub vid: VertexId,
}

impl<'s, Rt: RuntimeType> Error<'s, Rt> {
    pub fn new(node: ErrorNode<Rt>, at: SourceInfo<'s>, vid: VertexId) -> Self {
        Self { node, at, vid }
    }
}

pub struct TypeInferer<Rt: RuntimeType> {
    vertex_typ: BTreeMap<VertexId, type2::Typ<Rt>>,
    max_poly_deg: u64,
}

fn try_unwrap_poly_typ<'s, Rt: RuntimeType>(
    typ: &type2::Typ<Rt>,
    required_ptyp: PolyType,
    err: impl Fn(ErrorNode<Rt>) -> Error<'s, Rt>,
) -> Result<u64, Error<'s, Rt>> {
    let (ptyp, deg) = typ
        .try_unwrap_poly()
        .ok_or_else(|| err(ErrorNode::ExpectPolynomial))?;
    if *ptyp != required_ptyp {
        return Err(err(ErrorNode::ExpectPolynomialType(required_ptyp)));
    }
    Ok(*deg)
}

impl<Rt: RuntimeType> TypeInferer<Rt> {
    pub fn new() -> Self {
        Self {
            vertex_typ: BTreeMap::new(),
            max_poly_deg: 0,
        }
    }

    pub fn get_typ(&self, vid: VertexId) -> Option<&type2::Typ<Rt>> {
        self.vertex_typ.get(&vid)
    }

    fn try_unwrap_poly<'s, 'a>(
        &mut self,
        cg: &'a Cg<'s, Rt>,
        vid: VertexId,
        err: &impl Fn(ErrorNode<Rt>) -> Error<'s, Rt>,
    ) -> Result<(PolyType, u64), Error<'s, Rt>> {
        let (pty, deg) = self
            .infer(cg, vid)?
            .try_unwrap_poly()
            .cloned()
            .ok_or_else(|| err(ErrorNode::ExpectPolynomial))?;
        Ok((pty, deg))
    }

    fn try_unwrap_poly_typ<'s, 'a>(
        &mut self,
        cg: &'a Cg<'s, Rt>,
        vid: VertexId,
        required_ptyp: PolyType,
        err: impl Fn(ErrorNode<Rt>) -> Error<'s, Rt>,
    ) -> Result<u64, Error<'s, Rt>> {
        let typ = self.infer(cg, vid)?;
        let deg = try_unwrap_poly_typ(&typ, required_ptyp, err)?;
        Ok(deg)
    }

    fn try_unwrap_scalar<'s, 'a>(
        &mut self,
        cg: &'a Cg<'s, Rt>,
        vid: VertexId,
        err: &impl Fn(ErrorNode<Rt>) -> Error<'s, Rt>,
    ) -> Result<(), Error<'s, Rt>> {
        if let type2::Typ::Scalar = self.infer(cg, vid)? {
            Ok(())
        } else {
            Err(err(ErrorNode::ExpectScalar))
        }
    }

    fn try_unwrap_point_base<'s, 'a>(
        &mut self,
        cg: &'a Cg<'s, Rt>,
        vid: VertexId,
        err: &impl Fn(ErrorNode<Rt>) -> Error<'s, Rt>,
    ) -> Result<u32, Error<'s, Rt>> {
        let typ = self.infer(cg, vid)?;
        typ.try_unwrap_point_base()
            .ok_or_else(|| err(ErrorNode::ExpectPointBase))
    }

    fn try_unwrap_transcript<'s, 'a>(
        &mut self,
        cg: &'a Cg<'s, Rt>,
        vid: VertexId,
        err: &impl Fn(ErrorNode<Rt>) -> Error<'s, Rt>,
    ) -> Result<(), Error<'s, Rt>> {
        let typ = self.infer(cg, vid)?;
        if let type2::Typ::Transcript = typ {
            Ok(())
        } else {
            Err(err(ErrorNode::ExpectTranscript))
        }
    }

    fn infer_single_arith<'s>(
        &mut self,
        cg: &Cg<'s, Rt>,
        a: &Arith<VertexId>,
        err: &impl Fn(ErrorNode<Rt>) -> Error<'s, Rt>,
    ) -> Result<type2::Typ<Rt>, Error<'s, Rt>> {
        match a {
            Arith::Bin(BinOp::Pp(op), lhs, rhs) => {
                if op.support_coef() {
                    let (pty1, deg1) = self.try_unwrap_poly(cg, *lhs, err)?;
                    let (pty2, deg2) = self.try_unwrap_poly(cg, *rhs, err)?;
                    if pty1 != pty2 {
                        return Err(err(ErrorNode::ArithOnDifferentPolynomialTypes));
                    }
                    if deg1 != deg2 && pty1 == PolyType::Lagrange {
                        return Err(err(ErrorNode::ArithOnDifferentDegreeLagrangePolynomials(
                            deg1, deg2,
                        )));
                    }

                    Ok(type2::Typ::Poly((pty1, deg1)))
                } else {
                    let deg1 = self.try_unwrap_poly_typ(cg, *lhs, PolyType::Lagrange, err)?;
                    let deg2 = self.try_unwrap_poly_typ(cg, *rhs, PolyType::Lagrange, err)?;
                    Ok(type2::Typ::Poly((PolyType::Lagrange, deg1.max(deg2))))
                }
            }
            Arith::Bin(BinOp::Sp(op), lhs, rhs) => {
                if op.support_coef() {
                    self.try_unwrap_scalar(cg, *lhs, err)?;
                    let (pty, deg) = self.try_unwrap_poly(cg, *rhs, err)?;
                    Ok(type2::Typ::Poly((pty, deg)))
                } else {
                    self.try_unwrap_scalar(cg, *lhs, err)?;
                    let deg = self.try_unwrap_poly_typ(cg, *rhs, PolyType::Lagrange, err)?;
                    Ok(type2::Typ::Poly((PolyType::Lagrange, deg)))
                }
            }
            Arith::Bin(BinOp::Ss(..), lhs, rhs) => {
                self.try_unwrap_scalar(cg, *lhs, err)?;
                self.try_unwrap_scalar(cg, *rhs, err)?;
                Ok(type2::Typ::Scalar)
            }
            Arith::Unr(UnrOp::P(op), operand) => {
                if op.support_coef() {
                    let (pty, deg) = self.try_unwrap_poly(cg, *operand, err)?;
                    Ok(type2::Typ::Poly((pty, deg)))
                } else {
                    let deg = self.try_unwrap_poly_typ(cg, *operand, PolyType::Lagrange, err)?;
                    Ok(type2::Typ::Poly((PolyType::Lagrange, deg)))
                }
            }
            Arith::Unr(UnrOp::S(..), operand) => {
                self.try_unwrap_scalar(cg, *operand, err)?;
                Ok(type2::Typ::Scalar)
            }
        }
    }
    pub fn infer<'s>(
        &mut self,
        cg: &Cg<'s, Rt>,
        vid: VertexId,
    ) -> Result<type2::Typ<Rt>, Error<'s, Rt>> {
        if let Some(typ) = self.vertex_typ.get(&vid) {
            return Ok(typ.clone());
        }

        use type2::template::VertexNode::*;
        let v = cg.g.vertex(vid);
        let err = |node| Error::new(node, v.src().clone(), vid);

        let typ = match v.node() {
            NewPoly(deg, _, ptyp) => type2::Typ::Poly((*ptyp, *deg)),
            Constant(..) => v
                .try_to_type2_typ()
                .expect("a constant vertex should have complete type annotation from AST"),
            Extend(vin, to_deg) => {
                let (pty, deg) = self.try_unwrap_poly(cg, *vin, &err)?;
                if deg >= *to_deg {
                    return Err(err(ErrorNode::ExtendToSmallerDegree {
                        from: deg,
                        to: *to_deg,
                    }));
                }
                type2::Typ::Poly((pty, *to_deg))
            }
            SingleArith(a) => self.infer_single_arith(cg, a, &err)?,
            Arith { .. } => panic!("ArithGraph cannot come from AST"),
            Entry(..) => v
                .try_to_type2_typ()
                .expect("entry point of graph should have type annotation from AST"),
            Return(vin) => self.infer(cg, *vin)?,
            Ntt { s, to, from, .. } => {
                let deg = self.try_unwrap_poly_typ(cg, *s, *from, err)?;
                type2::Typ::Poly((*to, deg))
            }
            RotateIdx(vin, _) => {
                let deg = self.try_unwrap_poly_typ(cg, *vin, PolyType::Lagrange, err)?;
                type2::Typ::Poly((PolyType::Lagrange, deg))
            }
            Blind(vin, begin, end) => {
                let (pty, deg) = self.try_unwrap_poly(cg, *vin, &err)?;
                if !(*begin < *end && *end <= deg) {
                    return Err(err(ErrorNode::BadSlice {
                        begin: *begin,
                        end: *end,
                        deg,
                    }));
                }

                type2::Typ::Poly((pty, deg))
            }
            Slice(vin, begin, end) => {
                let (pty, deg) = self.try_unwrap_poly(cg, *vin, &err)?;
                if !(*begin < *end && *end <= deg) {
                    return Err(err(ErrorNode::BadSlice {
                        begin: *begin,
                        end: *end,
                        deg,
                    }));
                }

                type2::Typ::Poly((pty, *end - *begin))
            }
            Interpolate { xs, ys } => {
                if xs.len() != ys.len() {
                    return Err(err(ErrorNode::InterpolateArgsDifferentLength(
                        xs.len(),
                        ys.len(),
                    )));
                }
                for &x in xs {
                    self.try_unwrap_scalar(cg, x, &err)?;
                }
                for &y in ys {
                    self.try_unwrap_scalar(cg, y, &err)?;
                }

                type2::Typ::Poly((PolyType::Coef, xs.len() as u64))
            }
            Array(elements) => {
                let len = elements.len();

                let element_typ = elements
                    .iter()
                    .copied()
                    .enumerate()
                    .fold(Ok(None), |typ, (i, elem)| {
                        let typ = typ?;
                        let elem_typ = self.infer(cg, elem)?;
                        if let Some(typ) = typ {
                            if typ != elem_typ {
                                return Err(err(ErrorNode::ArrayOfInconsistentTypes(
                                    0, typ, i, elem_typ,
                                )));
                            }
                            Ok(Some(typ))
                        } else {
                            Ok(Some(elem_typ))
                        }
                    })?
                    .unwrap_or(type2::Typ::Scalar); // Default empty array to have element type scalar

                type2::Typ::Array(Box::new(element_typ), len)
            }
            AssmblePoly(deg, args) => {
                for &arg in args {
                    self.try_unwrap_scalar(cg, arg, &err)?;
                }

                type2::Typ::Poly((PolyType::Coef, *deg))
            }
            Msm { polys, points, alg } => {
                if points.len() != 1 {
                    panic!("MSM from AST should only have one set of base points");
                }

                let log_n = self.try_unwrap_point_base(cg, points[0], &err)?;
                let points_len = 2u64.pow(log_n);

                let _pty = polys
                    .iter()
                    .copied()
                    .enumerate()
                    .fold(Ok(None), |pty, (i, poly)| {
                        let pty = pty?;
                        let (poly_pty, poly_deg) = self.try_unwrap_poly(cg, poly, &err)?;
                        if poly_deg != points_len {
                            return Err(err(ErrorNode::MsmInconsistentInputLengths {
                                points_len,
                                poly_deg,
                                poly_i: i,
                            }));
                        }
                        if let Some(pty) = pty {
                            if pty != poly_pty {
                                return Err(err(ErrorNode::MsmInconsistentInputPolyTypes(
                                    0, pty, i, poly_pty,
                                )));
                            }
                            Ok(Some(pty))
                        } else {
                            Ok(Some(poly_pty))
                        }
                    })?
                    .unwrap();

                type2::Typ::Array(Box::new(type2::Typ::Point), polys.len())
            }
            HashTranscript {
                transcript, value, ..
            } => {
                self.try_unwrap_transcript(cg, *transcript, &err)?;
                let value_typ = self.infer(cg, *value)?;
                if !value_typ.hashable() {
                    return Err(err(ErrorNode::UnhashableToTranscript(value_typ)));
                }

                type2::Typ::Transcript
            }
            SqueezeScalar(transcript) => {
                self.try_unwrap_transcript(cg, *transcript, &err)?;
                type2::Typ::Tuple(vec![type2::Typ::Transcript, type2::Typ::Scalar])
            }
            TupleGet(tuple, idx) => {
                let tuple_typ = self.infer(cg, *tuple)?;
                let elem_types = tuple_typ
                    .try_unwrap_tuple()
                    .ok_or_else(|| err(ErrorNode::ExpectTuple))?;

                if *idx >= elem_types.len() {
                    return Err(err(ErrorNode::TupleIndexOutofBound {
                        index: *idx,
                        len: elem_types.len(),
                    }));
                }

                elem_types[*idx].clone()
            }
            ArrayGet(array, idx) => {
                let array_typ = self.infer(cg, *array)?;
                let (elem_type, array_len) = array_typ
                    .try_unwrap_array()
                    .ok_or_else(|| err(ErrorNode::ExpectArray))?;

                if *idx >= array_len {
                    return Err(err(ErrorNode::ArrayIndexOutofBound {
                        index: *idx,
                        len: array_len,
                    }));
                }

                elem_type.clone()
            }
            UserFunction(fid, ..) => {
                let f = &cg.user_function_table[*fid];
                f.ret_typ.clone()
            }
            KateDivision(poly, b) => {
                let deg = self.try_unwrap_poly_typ(cg, *poly, PolyType::Coef, &err)?;
                if deg < 1 {
                    return Err(err(ErrorNode::KateDivisionPolyDegreeTooSmall));
                }

                self.try_unwrap_scalar(cg, *b, &err)?;
                type2::Typ::Poly((PolyType::Coef, deg))
            }
            EvaluatePoly { poly, at: b } => {
                let _deg = self.try_unwrap_poly_typ(cg, *poly, PolyType::Coef, &err)?;
                self.try_unwrap_scalar(cg, *b, &err)?;
                type2::Typ::Scalar
            }
            BatchedInvert(poly) => {
                let deg = self.try_unwrap_poly_typ(cg, *poly, PolyType::Lagrange, &err)?;
                type2::Typ::Poly((PolyType::Lagrange, deg))
            }
            ScanMul { x0, poly } => {
                let deg = self.try_unwrap_poly_typ(cg, *poly, PolyType::Lagrange, &err)?;
                self.try_unwrap_scalar(cg, *x0, &err)?;
                type2::Typ::Poly((PolyType::Lagrange, deg))
            }
            DistributePowers { poly, powers } => {
                let (pty, deg) = self.try_unwrap_poly(cg, *poly, &err)?;
                let _powers_deg =
                    self.try_unwrap_poly_typ(cg, *powers, PolyType::Lagrange, &err)?;
                type2::Typ::Poly((pty, deg))
            }
            ScalarInvert { .. } => panic!("ScalarInvert cannot come from AST"),
            IndexPoly(poly, idx) => {
                let (_, deg) = self.try_unwrap_poly(cg, *poly, &err)?;

                if *idx >= deg {
                    return Err(err(ErrorNode::IndexPolyOutOfBound {
                        index: *idx,
                        len: deg,
                    }));
                }

                type2::Typ::Scalar
            }
            AssertEq(src, expected) => {
                let src_typ = self.infer(cg, *src)?;
                let expected_typ = self.infer(cg, *expected)?;
                if src_typ != expected_typ {
                    return Err(err(ErrorNode::AssertDifferentTypes(src_typ, expected_typ)));
                }
                src_typ
            }
            Print(x, _) => self.infer(cg, *x)?,
        };
        if let Some(annotated_typ) = v.typ() {
            if !annotated_typ.compatible_with_type2(&typ) {
                return Err(err(ErrorNode::IncompatibleWithAnnotation(
                    typ.clone(),
                    annotated_typ.clone(),
                )));
            }
        }

        if let Some((_, deg)) = typ.try_unwrap_poly() {
            self.max_poly_deg = self.max_poly_deg.max(*deg);
        }

        self.vertex_typ.insert(vid, typ.clone());

        Ok(typ)
    }

    pub fn max_poly_deg(&self) -> u64 {
        self.max_poly_deg
    }
}
