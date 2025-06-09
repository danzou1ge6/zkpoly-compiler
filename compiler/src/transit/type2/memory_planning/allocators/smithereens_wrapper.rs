use super::{normalize_size, regretting_integral, smithereens_allocator, OffsettedAddrMapping};
use crate::transit::type2::memory_planning::prelude::*;
use planning::machine::*;

enum PointsTo<P> {
    Inner(P),
    This,
}

pub struct SmithereensWrapp<'f, A, P, Rt: RuntimeType> {
    inner: A,
    sa: smithereens_allocator::Allocator,
    objects_at: Bijection<ObjectId, P>,
    pointers_to: Heap<P, PointsTo<P>>,
    _phantom: PhantomData<&'f Rt>,
}

pub type Procedure<'f, A, P, Rt: RuntimeType> = Box<
    dyn for<'a, 's, 'm, 'au, 'i> FnOnce(
            &mut Handle<'a, 'm, 's, 'au, 'i, A, P, Rt>,
        ) -> PlanningResponse<
            'f,
            's,
            ObjectId,
            P,
            Result<(), Error<'s>>,
            Rt,
        > + 'f,
>;

pub struct Handle<'a, 'm, 's, 'au, 'i, A, P, Rt: RuntimeType> {
    allocator: &'a mut SmithereensWrapp<'s, A, P, Rt>,
    machine: MachineHandle<'m, 's, ObjectId, P>,
    aux: &'au mut AuxiliaryInfo<'i, Rt>,
}

impl<'a, 'm, 's, 'au, 'i, A, P, Rt: RuntimeType> Handle<'a, 'm, 's, 'au, 'i, A, P, Rt>
where
    A: Allocator<'s, ObjectId, P, Rt>,
    P: 'static,
{
    fn inner_handle<'se, 'd>(&'se mut self) -> Box<dyn AllocatorHandle<'s, ObjectId, P, Rt> + 'd>
    where
        'se: 'd,
        Self: 'd,
    {
        // unsafe here is safe because this method's signature enforces that `self` (i.e. the handle)
        // is not useable until the returned handle is dropped.
        let machine = unsafe { std::ptr::read(&mut self.machine) };
        self.allocator.inner.handle(machine, self.aux)
    }
}

// impl<'a, 'm, 's, 'au, 'i, A, P, Rt: RuntimeType> AllocatorHandle<'s, ObjectId, P, Rt>
//     for Handle<'a, 'm, 's, 'au, 'i, A, P, Rt>
// where
//     A: Allocator<'s, ObjectId, P, Rt>,
//     P: UsizeId + 'static,
// {
//     fn access(&mut self, t: &ObjectId) -> Option<P> {
//         if let Some(p) = self.allocator.objects_at.get_forward(t) {
//             Some(*p)
//         } else {
//             self.inner_handle().access(t)
//         }
//     }

//     fn allocate(&mut self, size: Size, t: &ObjectId) -> allocator::AResp<'s, ObjectId, P, P, Rt> {
//         let size = super::normalize_size(size);

//         match size {
//             Size::Integral(is) => {
//                 let resp = self.inner_handle().allocate(size, t);
//                 resp.bind_result({
//                     let after_inner =
//                         self.add_procedure(Box::new(|handle| {
//                             Response::Complete(Ok(()))
//                         }));
//                     move |_| after_inner
//                 })
//             }
//             _ => todo!(),
//         };

//         todo!()
//     }
// }
