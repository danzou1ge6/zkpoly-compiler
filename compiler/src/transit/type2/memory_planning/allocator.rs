use crate::transit::type2::memory_planning::prelude::*;

pub struct Completeness(u64, u64);

impl Completeness {
    pub fn new(numerator: u64, denominator: u64) -> Self {
        Self(numerator, denominator)
    }

    pub fn is_one(&self) -> bool {
        self.0 == self.1
    }

    pub fn non_zero(&self) -> bool {
        self.0 != 0
    }

    pub fn plain_one() -> Self {
        Self(1, 1)
    }

    pub fn plain_zero() -> Self {
        Self(0, 1)
    }
}

/// Handle to an allocator, with which memory can be manipulated.
///
/// See [`Allocator`] for what [`P`] is for.
pub trait AllocatorHandle<'s, T, P, Rt: RuntimeType> {
    fn device(&self) -> Device;

    /// Allocate on memory device some token.
    /// Returns the allocated pointer, but this pointer may be immediately invalidated after some other operations
    /// such as `read` or `allocate`.
    ///
    /// Token must not be recorded on device.
    fn allocate<'f>(
        &mut self,
        size: Size,
        t: &T,
    ) -> Response<'f, planning::Machine<'s, T, P>, T, P, Result<P, Error<'s>>, Rt>
    where
        's: 'f;

    /// Deallocate the token, which must be recorded on device, otherwise panics.
    /// That is, it can be ejected, but it must has been allocated.
    fn deallocate<'f>(&mut self, t: &T) -> Response<'f, planning::Machine<'s, T, P>, T, P, (), Rt>
    where
        's: 'f;

    /// Reuse space for object.
    /// The object must be accessible, otherwise panics.
    fn reuse(&mut self, new_object: ObjectId, old_object: ObjectId);

    /// Ask the allocator to claim token from some device.
    fn claim<'f>(
        &mut self,
        t: &T,
        size: Size,
        from: Device,
    ) -> Response<'f, planning::Machine<'s, T, P>, T, P, Result<(), Error<'s>>, Rt>
    where
        's: 'f;

    /// Get reference of pointer which can be used to access this token.
    /// The token must be recorded on device.
    ///
    /// If the token can't be accessed, return None.
    ///
    /// We need mutable access here because next-use may be updated.
    fn access(&mut self, t: &T) -> Option<P>;

    /// Get completeness of the object on this device.
    /// Completeness may be a number between zero and one if the device uses page allocation
    /// and ejects by pages.
    fn completeness(&self, object: ObjectId) -> Completeness;
}

pub trait AllocatorRealizer<'s, T, P, Rt: RuntimeType> {
    /// Instruct the machine to allocate space at pointer.
    /// Relative registers should be made available after this.
    fn allocate(&mut self, t: &T, pointer: &P);

    /// Instruct the machine to deallocate space at pointer
    fn deallocate(&mut self, t: &T, pointer: &P);
    
    /// Instruct the machine to perform a transfer
    fn transfer<'f>(
        &mut self,
        t: &T,
        from_pointer: &P,
        to_device: Device,
        to_pointer: &P,
    ) -> RealizationResponse<'f, 's, T, P, Result<(), Error<'s>>, Rt>;
}

/// A memory allocator.
///
/// # Type Parameters
/// [`P`] is how memory planner keep tracks of pointers.
/// [`T`] is smallest unit of data involved in memory management.
///
/// Allocator is only responsible for keeping track of the memory device's
/// internal state, but not how the device is manipulated at runtime.
/// Therefore, we need a handle with mutable reference to [`Machine`],
/// and [`Machine`] keeps track of the opertions needed to manipulate the device.
///
/// After all memory devices have been planned, we need to convert the resulting operation
/// sequence to Type3, which is done by the realizer.
/// Realizer is responsible for emitting currespounding allocation/deallocation/transfer Type3
/// instructions.
pub trait Allocator<T, P, Rt: RuntimeType> {
    fn handle<'a, 'b, 'c, 'd, 's, 'i>(
        &'a mut self,
        machine: planning::MachineHandle<'b, 's, T, P>,
        aux: &'c mut AuxiliaryInfo<'i, Rt>,
    ) -> Box<dyn AllocatorHandle<'s, T, P, Rt> + 'd>
    where
        'a: 'd,
        'b: 'd,
        'c: 'd,
        'i: 'd;

    fn realizer<'a, 'b, 'c, 'd, 's, 'i>(
        &'a mut self,
        machine: realization::MachineHandle<'b, 's, P>,
        aux: &'c mut AuxiliaryInfo<'i, Rt>,
    ) -> Box<dyn AllocatorRealizer<'s, T, P, Rt> + 'd>
    where
        'a: 'd,
        'b: 'd,
        'c: 'd,
        'i: 'd;
}

pub struct AllocatorCollection<'a, T, P, Rt: RuntimeType>(BTreeMap<Device, &'a mut (dyn Allocator<T, P, Rt> + 'static)>);

impl<'a, T, P, Rt: RuntimeType> AllocatorCollection<'a, T, P, Rt> {
    pub fn get(&mut self, device: Device) -> &mut dyn Allocator<T, P, Rt> {
        *self.0.get_mut(&device).unwrap()
    }

    pub fn handle<'b, 'm, 's, 'aux, 'c, 'i>(
        &'b mut self,
        device: Device,
        machine: &'m mut planning::Machine<'s, T, P>,
        aux: &'aux mut AuxiliaryInfo<'i, Rt>,
    ) -> Box<dyn AllocatorHandle<'s, T, P, Rt> + 'c>
    where
        'm: 'c,
        'a: 'c,
        'aux: 'c,
        'b: 'c,
    {
        self.get(device).handle(machine.handle(device), aux)
    }

    pub fn realizer<'b, 'm, 's, 'aux, 'c, 'i>(
        &'b mut self,
        device: Device,
        machine: &'m mut realization::Machine<'s, P>,
        aux: &'aux mut AuxiliaryInfo<'i, Rt>,
    ) -> Box<dyn AllocatorRealizer<'s, T, P, Rt> + 'c>
    where
        'm: 'c,
        'a: 'c,
        'aux: 'c,
        'b: 'c,
    {
        self.get(device).realizer(machine.handle(device), aux)
    }

    pub fn insert<'b>(
        self,
        device: Device,
        allocator: &'b mut (dyn Allocator<T, P, Rt> + 'static),
    ) -> AllocatorCollection<'b, T, P, Rt>
    where
        'a: 'b,
    {
        let mut map: BTreeMap<Device, &'b mut (dyn Allocator<T, P, Rt> + 'static)> = self.0;
        map.insert(device, allocator);
        AllocatorCollection(map)
    }

    pub fn object_available_on<'s, 'i>(
        &mut self,
        devices: impl Iterator<Item = Device>,
        object: ObjectId,
        machine: &mut planning::Machine<'s, T, P>,
        aux: &mut AuxiliaryInfo<'i, Rt>,
    ) -> Vec<Device> {
        devices
            .filter(move |&device| {
                self.handle(device, machine, aux)
                    .completeness(object)
                    .non_zero()
            })
            .collect()
    }
}

impl<'a, T, P, Rt: RuntimeType> FromIterator<(Device, &'a mut (dyn Allocator<T, P, Rt> + 'static))>
    for AllocatorCollection<'a, T, P, Rt>
{
    fn from_iter<It: IntoIterator<Item = (Device, &'a mut (dyn Allocator<T, P, Rt> + 'static))>>(
        iter: It,
    ) -> Self {
        Self(iter.into_iter().collect())
    }
}
