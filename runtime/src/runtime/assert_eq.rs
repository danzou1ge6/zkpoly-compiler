use crate::args::{RuntimeType, Variable};

pub fn assert_eq<Rt: RuntimeType>(value: &Variable<Rt>, expected: &Variable<Rt>) -> bool {
    match (value, expected) {
        (Variable::Scalar(a), Variable::Scalar(b)) => {
            assert!(a.device.is_cpu());
            assert!(b.device.is_cpu());
            *a.as_ref() == *b.as_ref()
        }
        (Variable::Any(_), Variable::Any(_)) => unreachable!("any can't be compared"),
        (Variable::GpuBuffer(_), Variable::GpuBuffer(_)) => {
            unreachable!("gpu buffer can't be compared")
        }
        (Variable::Point(a), Variable::Point(b)) => *a.as_ref() == *b.as_ref(),
        (Variable::PointArray(a), Variable::PointArray(b)) => {
            assert!(a.device.is_cpu());
            assert!(b.device.is_cpu());
            a == b
        }
        (Variable::ScalarArray(a), Variable::ScalarArray(b)) => {
            assert!(a.device.is_cpu());
            assert!(b.device.is_cpu());
            a == b
        }
        (Variable::Stream(_), Variable::Stream(_)) => unreachable!("stream can't be compared"),
        (Variable::Tuple(a), Variable::Tuple(b)) => {
            if a.len() != b.len() {
                return false;
            }
            for (a, b) in a.iter().zip(b.iter()) {
                if !assert_eq(a, b) {
                    return false;
                }
            }
            true
        }
        (Variable::Transcript(_), Variable::Transcript(_)) => {
            unimplemented!("transcript doesn't implement PartialEq")
        }
        _ => false,
    }
}
