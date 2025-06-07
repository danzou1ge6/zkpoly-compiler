// functions in this file is mainly copied from halo2, we only provide a wrapper for the runtime

use std::ptr;

use group::ff::{BatchInvert, Field};
use zkpoly_runtime::{
    args::{RuntimeType, Variable},
    error::RuntimeError,
    functions::{FuncMeta, Function, FunctionValue, KernelType, RegisteredFunction},
    scalar::ScalarArray,
    transcript::{ChallengeScalarUnit, Transcript, TranscriptWrite},
};

#[macro_export]
macro_rules! define_cpu_kernel {
    ($name:ident) => {
        pub struct $name<T: RuntimeType> {
            _marker: std::marker::PhantomData<T>,
        }

        impl<T: RuntimeType> $name<T> {
            pub fn new() -> Self {
                Self {
                    _marker: std::marker::PhantomData,
                }
            }
        }
    };
}

// Usage example:
define_cpu_kernel!(InterpolateKernel);
define_cpu_kernel!(AssmblePoly);
define_cpu_kernel!(HashTranscript);
define_cpu_kernel!(HashTranscriptWrite);
define_cpu_kernel!(SqueezeScalar);

/// Returns coefficients of an n - 1 degree polynomial given a set of n points
/// and their evaluations. This function will panic if two values in `points`
/// are the same.
pub fn lagrange_interpolate<F: Field>(points: &Vec<F>, evals: &Vec<F>, res: &mut ScalarArray<F>) {
    assert_eq!(points.len(), evals.len());
    assert_eq!(points.len(), res.len());
    if points.len() == 1 {
        // Constant polynomial
        res[0] = evals[0];
    } else {
        let mut denoms = Vec::with_capacity(points.len());
        for (j, x_j) in points.iter().enumerate() {
            let mut denom = Vec::with_capacity(points.len() - 1);
            for x_k in points
                .iter()
                .enumerate()
                .filter(|&(k, _)| k != j)
                .map(|a| a.1)
            {
                denom.push(*x_j - x_k);
            }
            denoms.push(denom);
        }
        // Compute (x_j - x_k)^(-1) for each j != i
        denoms.iter_mut().flat_map(|v| v.iter_mut()).batch_invert();
        unsafe {
            ptr::write_bytes(res.values.get_ptr().0, 0x00, res.len());
        }
        for (j, (denoms, eval)) in denoms.into_iter().zip(evals.iter()).enumerate() {
            let mut tmp: Vec<F> = Vec::with_capacity(points.len());
            let mut product = Vec::with_capacity(points.len() - 1);
            tmp.push(F::ONE);
            for (x_k, denom) in points
                .iter()
                .enumerate()
                .filter(|&(k, _)| k != j)
                .map(|a| a.1)
                .zip(denoms.into_iter())
            {
                product.resize(tmp.len() + 1, F::ZERO);
                for ((a, b), product) in tmp
                    .iter()
                    .chain(std::iter::once(&F::ZERO))
                    .zip(std::iter::once(&F::ZERO).chain(tmp.iter()))
                    .zip(product.iter_mut())
                {
                    *product = *a * (-denom * x_k) + *b * denom;
                }
                std::mem::swap(&mut tmp, &mut product);
            }
            assert_eq!(tmp.len(), points.len());
            assert_eq!(product.len(), points.len() - 1);
            for (final_coeff, interpolation_coeff) in res.iter_mut().zip(tmp.into_iter()) {
                *final_coeff += interpolation_coeff * eval;
            }
        }
    }
}

impl<T: RuntimeType> RegisteredFunction<T> for InterpolateKernel<T> {
    fn get_fn(&self) -> Function<T> {
        let rust_func = |mut mut_var: Vec<&mut Variable<T>>,
                         var: Vec<&Variable<T>>|
         -> Result<(), RuntimeError> {
            assert_eq!(mut_var.len(), 1);
            let res = mut_var[0].unwrap_scalar_array_mut();

            assert_eq!(var.len(), res.len() * 2);
            let points = (0..res.len())
                .into_iter()
                .map(|i| *var[i].unwrap_scalar().as_ref())
                .collect::<Vec<_>>();
            let evals = (res.len()..2 * res.len())
                .into_iter()
                .map(|i| *var[i].unwrap_scalar().as_ref())
                .collect::<Vec<_>>();
            lagrange_interpolate(&points, &evals, res);
            Ok(())
        };
        Function {
            meta: FuncMeta::new("interpolate".to_string(), KernelType::Interpolate),
            f: FunctionValue::Fn(Box::new(rust_func)),
        }
    }
}

impl<T: RuntimeType> RegisteredFunction<T> for AssmblePoly<T> {
    fn get_fn(&self) -> Function<T> {
        let rust_func = |mut mut_var: Vec<&mut Variable<T>>,
                         var: Vec<&Variable<T>>|
         -> Result<(), RuntimeError> {
            assert_eq!(mut_var.len(), 1);
            let res = mut_var[0].unwrap_scalar_array_mut();
            assert_eq!(var.len(), res.len());
            for (res, var) in res.iter_mut().zip(var.iter()) {
                *res = var.unwrap_scalar().as_ref().clone();
            }
            Ok(())
        };
        Function {
            meta: FuncMeta::new("assmble_poly".to_string(), KernelType::AssmblePoly),
            f: FunctionValue::Fn(Box::new(rust_func)),
        }
    }
}

impl<T: RuntimeType> RegisteredFunction<T> for HashTranscript<T> {
    fn get_fn(&self) -> Function<T> {
        let rust_func = |mut mut_var: Vec<&mut Variable<T>>,
                         var: Vec<&Variable<T>>|
         -> Result<(), RuntimeError> {
            assert_eq!(mut_var.len(), 1);
            assert_eq!(var.len(), 1);
            let res = mut_var[0].unwrap_transcript_mut().as_mut();
            match var[0] {
                Variable::ScalarArray(scalar_array) => {
                    for scalar in scalar_array.iter() {
                        res.common_scalar(scalar.clone().into()).unwrap();
                    }
                }
                Variable::Scalar(scalar) => {
                    res.common_scalar(scalar.as_ref().clone().into()).unwrap();
                }
                Variable::Point(point) => {
                    res.common_point(point.as_ref().clone()).unwrap();
                }
                _ => unreachable!("only implmented for poly, scalar and point"),
            }
            Ok(())
        };
        Function {
            meta: FuncMeta::new("hash_transcript".to_string(), KernelType::HashTranscript),
            f: FunctionValue::Fn(Box::new(rust_func)),
        }
    }
}

impl<T: RuntimeType> RegisteredFunction<T> for HashTranscriptWrite<T> {
    fn get_fn(&self) -> Function<T> {
        let rust_func = |mut mut_var: Vec<&mut Variable<T>>,
                         var: Vec<&Variable<T>>|
         -> Result<(), RuntimeError> {
            assert_eq!(mut_var.len(), 1);
            assert_eq!(var.len(), 1);
            let res = mut_var[0].unwrap_transcript_mut().as_mut();
            match var[0] {
                Variable::ScalarArray(scalar_array) => {
                    for scalar in scalar_array.iter() {
                        res.write_scalar(scalar.clone().into()).unwrap();
                    }
                }
                Variable::Scalar(scalar) => {
                    res.write_scalar(scalar.as_ref().clone().into()).unwrap();
                }
                Variable::Point(point) => {
                    res.write_point(point.as_ref().clone()).unwrap();
                }
                _ => unreachable!("only implmented for poly, scalar and point"),
            }
            Ok(())
        };
        Function {
            meta: FuncMeta::new(
                "hash_transcript_write".to_string(),
                KernelType::HashTranscriptWrite,
            ),
            f: FunctionValue::Fn(Box::new(rust_func)),
        }
    }
}

impl<T: RuntimeType> RegisteredFunction<T> for SqueezeScalar<T> {
    fn get_fn(&self) -> Function<T> {
        let rust_func = |mut mut_var: Vec<&mut Variable<T>>,
                         var: Vec<&Variable<T>>|
         -> Result<(), RuntimeError> {
            assert_eq!(mut_var.len(), 2);
            assert_eq!(var.len(), 0);
            let (trans_buf, scalar_buf) = mut_var.split_at_mut(1);
            let transcript = trans_buf[0].unwrap_transcript_mut().as_mut();
            let scalar = scalar_buf[0].unwrap_scalar_mut();
            let c_scalar: ChallengeScalarUnit<_> = transcript.squeeze_challenge_scalar();
            *scalar.as_mut() = T::Field::from(*c_scalar);
            Ok(())
        };
        Function {
            meta: FuncMeta::new("squeeze_scalar".to_string(), KernelType::SqueezeScalar),
            f: FunctionValue::Fn(Box::new(rust_func)),
        }
    }
}
