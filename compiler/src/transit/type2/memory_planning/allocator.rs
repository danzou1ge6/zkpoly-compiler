use crate::transit::type2::memory_planning::prelude::*;

pub struct Cpu;
pub struct Gpu;
pub struct Disk;

pub trait DeviceMarker: 'static {}

impl DeviceMarker for Cpu {}
impl DeviceMarker for Gpu {}
impl DeviceMarker for Disk {}

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

pub type AResp<'s, T, P, R, Rt: RuntimeType> =
    Response<'s, planning::Machine<'s, T, P>, T, P, Result<R, Error<'s>>, Rt>;

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
    fn allocate(&mut self, size: Size, t: &T) -> AResp<'s, T, P, P, Rt>;

    /// Deallocate the token, which must be recorded on device, otherwise panics.
    fn deallocate(&mut self, t: &T) -> AResp<'s, T, P, (), Rt>;

    /// Reuse space for object.
    /// The object must be accessible, otherwise panics.
    fn reuse(&mut self, new_object: ObjectId, old_object: ObjectId);

    /// Ask the allocator to claim token from some device.
    fn claim(&mut self, t: &T, size: Size, from: Device) -> AResp<'s, T, P, (), Rt>;

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
    fn completeness(&mut self, object: ObjectId) -> Completeness;

    fn typeid(&self) -> typeid::ConstTypeId;
}

pub trait AllocatorRealizer<'s, T, P, Rt: RuntimeType> {
    /// Instruct the machine to allocate space at pointer.
    /// Relative registers should be made available after this.
    fn allocate(&mut self, t: &T, pointer: &P);

    /// Instruct the machine to deallocate space at pointer
    fn deallocate(&mut self, t: &T, pointer: &P);

    /// Instruct the machine to perform a transfer
    fn transfer(
        &mut self,
        t: &T,
        from_pointer: &P,
        to_device: Device,
        to_pointer: &P,
    ) -> RealizationResponse<'s, T, P, Result<(), Error<'s>>, Rt>;
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
pub trait Allocator<'s, T, P, Rt: RuntimeType> {
    fn handle<'a, 'b, 'c, 'd, 'i>(
        &'a mut self,
        machine: planning::MachineHandle<'b, 's, T, P>,
        aux: &'c mut AuxiliaryInfo<'i, Rt>,
    ) -> Box<dyn AllocatorHandle<'s, T, P, Rt> + 'd>
    where
        'a: 'd,
        'b: 'd,
        'c: 'd,
        'i: 'd;

    fn realizer<'a, 'b, 'c, 'd, 'i>(
        &'a mut self,
        machine: realization::MachineHandle<'b, 's, P>,
        aux: &'c mut AuxiliaryInfo<'i, Rt>,
    ) -> Box<dyn AllocatorRealizer<'s, T, P, Rt> + 'd>
    where
        'a: 'd,
        'b: 'd,
        'c: 'd,
        'i: 'd;

    fn allcate_pointer(&mut self) -> P;

    fn inner<'t>(&'t mut self) -> Option<&'t mut dyn Allocator<'s, T, P, Rt>>;
}

pub struct AllocatorCollection<'a, 's, T, P, Rt: RuntimeType>(
    BTreeMap<Device, &'a mut (dyn Allocator<'s, T, P, Rt> + 's)>,
);

impl<'a, 's, T, P, Rt: RuntimeType> AllocatorCollection<'a, 's, T, P, Rt> {
    pub fn get(&mut self, device: Device) -> &mut dyn Allocator<'s, T, P, Rt> {
        *self.0.get_mut(&device).unwrap()
    }

    pub fn handle<'b, 'm, 'aux, 'c, 'i>(
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

    pub fn realizer<'b, 'm, 'aux, 'c, 'i>(
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
        allocator: &'b mut (dyn Allocator<'s, T, P, Rt> + 's),
    ) -> AllocatorCollection<'b, 's, T, P, Rt>
    where
        'a: 'b,
    {
        let mut map: BTreeMap<Device, &'b mut (dyn Allocator<'s, T, P, Rt> + 's)> = self.0;
        map.insert(device, allocator);
        AllocatorCollection(map)
    }

    pub fn object_available_on<'i>(
        &mut self,
        devices: impl Iterator<Item = Device>,
        object: ObjectId,
        machine: &mut planning::Machine<'s, T, P>,
        aux: &mut AuxiliaryInfo<'i, Rt>,
    ) -> Vec<Device> {
        let devices = devices
            .filter(move |&device| {
                self.handle(device, machine, aux)
                    .completeness(object)
                    .non_zero()
            })
            .collect();

        // fixme
        println!("{:?} is available on {:?}", object, devices);

        devices
    }
}

impl<'a, 's, T, P, Rt: RuntimeType>
    FromIterator<(Device, &'a mut (dyn Allocator<'s, T, P, Rt> + 's))>
    for AllocatorCollection<'a, 's, T, P, Rt>
{
    fn from_iter<It: IntoIterator<Item = (Device, &'a mut (dyn Allocator<'s, T, P, Rt> + 's))>>(
        iter: It,
    ) -> Self {
        Self(iter.into_iter().collect())
    }
}
