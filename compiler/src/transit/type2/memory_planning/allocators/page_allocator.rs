use std::u64;

use super::super::prelude::*;

mod pages {
    use zkpoly_common::segment_tree::SegmentTree;

    use super::*;

    define_usize_id!(PageId);

    /// A set of pages with their next_use tracked.
    pub struct Pages {
        occupied: Heap<PageId, bool>,
        free_pages: BTreeSet<PageId>,
        // Invariant: next_use[page] = 0 if page is free.
        // This way, the page with maximum next_use is always a occuped page,
        // if any page is occupied at all.
        next_use: SegmentTree<u64>,
    }

    impl Pages {
        pub fn new(number_pages: usize) -> Self {
            Self {
                occupied: Heap::repeat(false, number_pages),
                next_use: SegmentTree::new(&vec![0; number_pages]),
                free_pages: (0..number_pages).map(PageId::from).collect(),
            }
        }

        fn debug(&mut self) {
            println!("Pages:");
            for (i, occupied) in self.occupied.iter_with_id() {
                let next_use = self
                    .next_use
                    .query_max_info(usize::from(i), usize::from(i))
                    .map(|x| x.value);
                println!("page {:?}: occupied={}, next_use={:?}", i, occupied, next_use);
            }
        }

        /// Mark a page as occupied. The page cannot already be occupied.
        pub fn occupy(&mut self, page: PageId, next_use: Option<Index>) {
            if self.occupied[page] {
                panic!("page {} already occupied", usize::from(page));
            }
            self.occupied[page] = true;
            self.free_pages.remove(&page);

            let page = usize::from(page);
            let next_use = next_use.map_or(u64::MAX, |v| usize::from(v) as u64);
            self.next_use.modify_set(page, page, next_use);
        }

        /// Free a page, returning its next_use. The page cannot already be free.
        pub fn free(&mut self, page: PageId) -> u64 {
            if !self.occupied[page] {
                panic!("page {} already free", usize::from(page));
            }

            self.occupied[page] = false;
            self.free_pages.insert(page);

            let page = usize::from(page);
            let next_use = self.next_use.query_max_info(page, page).unwrap().value;

            self.next_use.modify_set(page, page, 0);

            next_use
        }

        /// Find the page with maximum next_use.
        /// Returns none if no page is occupied.
        pub fn decide_victim(&mut self, pc: Index) -> Option<PageId> {
            let query_result = self
                .next_use
                .query_max_info(0, self.occupied.len() - 1)
                .unwrap();
            let index = PageId::from(query_result.index);

            if query_result.value <= usize::from(pc) as u64 {
                self.debug();
                panic!("deciding victim {:?} to be less than pc {:?}", index, pc);
            }

            if self.occupied[index] {
                Some(index)
            } else {
                None
            }
        }

        pub fn update_next_use(&mut self, page: PageId, to: Index) {
            let page = usize::from(page);
            let to = usize::from(to);
            self.next_use.modify_set(page, page, to as u64);
        }

        /// Return a vector of `number` free pages, or none if not enough free pages
        pub fn try_find_free_pages(&mut self, number: usize) -> Option<Vec<PageId>> {
            if self.free_pages.len() >= number {
                Some(
                    (0..number)
                        .map(|_| self.free_pages.pop_first())
                        .collect::<Option<Vec<_>>>()
                        .expect("we have confirmed that there are enough free pages"),
                )
            } else {
                None
            }
        }

        pub fn number(&self) -> usize {
            self.occupied.len()
        }

        pub fn occupancy(&self) -> f32 {
            1.0 - self.free_pages.len() as f32 / self.occupied.len() as f32
        }
    }
}

use pages::{PageId, Pages};

pub struct PageAllocator<'f, P, Rt: RuntimeType, D: DeviceMarker> {
    page_size: u64,
    pages: Pages,
    living_objects: BTreeMap<ObjectId, (P, Size)>,
    page_mapping: Heap<P, (Vec<PageId>, Size)>,
    page_objects: Heap<PageId, Option<ObjectId>>,
    _phantom: PhantomData<(&'f Rt, D)>,
}

impl<'f, P, Rt: RuntimeType, D: DeviceMarker> PageAllocator<'f, P, Rt, D> {
    pub fn new(number_pages: usize, page_size: u64) -> Self {
        Self {
            page_size,
            pages: Pages::new(number_pages),
            living_objects: BTreeMap::new(),
            page_mapping: Heap::new(),
            page_objects: Heap::repeat(None, number_pages),
            _phantom: PhantomData,
        }
    }
}

pub struct Handle<'a, 'm, 's, 'au, 'i, P, Rt: RuntimeType, D: DeviceMarker> {
    allocator: &'a mut PageAllocator<'s, P, Rt, D>,
    machine: planning::MachineHandle<'m, 's, ObjectId, P>,
    aux: &'au mut AuxiliaryInfo<'i, Rt>,
}

impl<'a, 'm, 's, 'au, 'i, P, Rt: RuntimeType, D: DeviceMarker> Handle<'a, 'm, 's, 'au, 'i, P, Rt, D>
where
    P: UsizeId + 'static,
{
    fn allocate_pages(
        &mut self,
        pages: Vec<PageId>,
        object: ObjectId,
        next_use: Option<Index>,
        size: Size,
    ) -> P {
        pages
            .iter()
            .copied()
            .for_each(|page| self.allocator.page_objects[page] = Some(object));
        pages
            .iter()
            .copied()
            .for_each(|page| self.allocator.pages.occupy(page, next_use));

        let p = self.allocator.page_mapping.push((pages, size));
        self.allocator.living_objects.insert(object, (p, size));

        self.machine.allocate(object, p);
        p
    }

    fn deallocate_pages(&mut self, victim_object: ObjectId) -> (P, Size) {
        let (victim_pointer, victim_size) = self
            .allocator
            .living_objects
            .remove(&victim_object)
            .unwrap_or_else(|| panic!("object {:?} not allocated", victim_object));
        let (victim_pages, _) = self.allocator.page_mapping[victim_pointer].clone();

        victim_pages.iter().for_each(|page| {
            self.allocator.pages.free(*page);
            self.allocator.page_objects[*page] = None;
        });

        self.machine.deallocate(victim_object, victim_pointer);

        (victim_pointer, victim_size)
    }

    fn update_pages_next_use(&mut self, p: P, next_use: Index) {
        let (pages, _) = &self.allocator.page_mapping[p];
        pages.iter().for_each(|page| {
            self.allocator.pages.update_next_use(*page, next_use);
        });
    }
}

impl<'a, 'm, 's, 'au, 'i, P, Rt: RuntimeType, D: DeviceMarker> AllocatorHandle<'s, ObjectId, P, Rt>
    for Handle<'a, 'm, 's, 'au, 'i, P, Rt, D>
where
    P: UsizeId + 'static,
{
    fn access(&mut self, t: &ObjectId) -> Option<P> {
        let next_use = self
            .aux
            .mark_use(t.clone(), self.device())
            .unwrap_or(Index::inf());

        self.allocator
            .living_objects
            .get(t)
            .map(|(a, _)| a)
            .cloned()
            .map(|p| {
                self.update_pages_next_use(p, next_use);
                p
            })
    }

    fn allocate(&mut self, size: Size, t: &ObjectId) -> allocator::AResp<'s, ObjectId, P, P, Rt> {
        if let Some(_) = self.allocator.living_objects.get(t) {
            panic!("{:?} already allocated", t);
        }

        let number = u64::from(size).div_ceil(self.allocator.page_size) as usize;
        let next_use = self.aux.query_next_use(t.clone(), self.device());

        if number > self.allocator.pages.number() {
            return Response::Complete(Err(Error::VertexInputsOutputsNotAccommodated(None)));
        }

        if let Some(pages) = self.allocator.pages.try_find_free_pages(number) {
            // fixme
            println!("{:?} allocating {} pages for {:?} with size {:?}", self.device(), pages.len(), t, size);

            let p = self.allocate_pages(pages, *t, next_use, size);
            Response::Complete(Ok(p))
        } else {
            let victim_page = self
                .allocator
                .pages
                .decide_victim(self.aux.pc())
                .expect("since there are occupied pages, a victim page must can be found");
            let victim_object = self.allocator.page_objects[victim_page]
                .expect("an occupied page must belong to some object");

            let (victim_pointer, victim_size) = self
                .allocator
                .living_objects
                .get(&victim_object)
                .cloned()
                .unwrap_or_else(|| panic!("object {:?} not allocated", victim_object));
            let t = *t;
            let this_device = self.device();

            // fixme
            println!("{:?} ejecting {:?}", this_device, victim_object);

            Response::Continue(
                Continuation::simple_eject(
                    self.device(),
                    victim_pointer,
                    victim_object,
                    victim_size,
                )
                .bind_result(move |_| {
                    Continuation::new(move |allocators, machine, aux| {
                        let resp = allocators
                            .handle(this_device, machine, aux)
                            .deallocate(&victim_object);
                        resp.commit(allocators, machine, aux)?;

                        let resp = allocators
                            .handle(this_device, machine, aux)
                            .allocate(size, &t);
                        resp.commit(allocators, machine, aux)
                    })
                }),
            )
        }
    }

    fn claim(
        &mut self,
        t: &ObjectId,
        size: Size,
        from: Device,
    ) -> allocator::AResp<'s, ObjectId, P, (), Rt> {
        if self.allocator.living_objects.contains_key(t) {
            return Response::Complete(Ok(()));
        }

        let this_device = self.device();
        let object = t.clone();

        self.allocate(size, t)
            .bind_result(move |p| Continuation::simple_provide(this_device, p, from, object))
    }

    fn completeness(&mut self, object: ObjectId) -> Completeness {
        if self.allocator.living_objects.contains_key(&object) {
            Completeness::plain_one()
        } else {
            Completeness::plain_zero()
        }
    }

    fn deallocate(&mut self, t: &ObjectId) -> allocator::AResp<'s, ObjectId, P, (), Rt> {
        let _ = self.deallocate_pages(*t);
        Response::ok(())
    }

    fn device(&self) -> Device {
        self.machine.device()
    }

    fn reuse(&mut self, new_object: ObjectId, old_object: ObjectId) {
        if let Some((p, size)) = self.allocator.living_objects.remove(&old_object) {
            let next_use = self
                .aux
                .query_next_use(new_object.clone(), self.device())
                .unwrap_or(Index::inf());
            self.update_pages_next_use(p, next_use);
            self.allocator.living_objects.insert(new_object, (p, size));

            let (pages, _) = &self.allocator.page_mapping[p];
            pages.iter().for_each(|page| {
                self.allocator.page_objects[*page] = Some(new_object);
            })
        } else {
            panic!("object {:?} not allocated", old_object)
        }
    }

    fn typeid(&self) -> typeid::ConstTypeId {
        typeid::ConstTypeId::of::<Self>()
    }
}

pub struct Realizer<'a, 'm, 's, 'au, 'i, 'f, P, Rt: RuntimeType, D: DeviceMarker> {
    allocator: &'a mut PageAllocator<'f, P, Rt, D>,
    machine: realization::MachineHandle<'m, 's, P, Rt>,
    aux: &'au AuxiliaryInfo<'i, Rt>,
}

impl<'a, 'm, 's, 'au, 'i, 'f, P, Rt: RuntimeType, D: DeviceMarker>
    AllocatorRealizer<'s, ObjectId, P, Rt> for Realizer<'a, 'm, 's, 'au, 'i, 'f, P, Rt, D>
where
    P: UsizeId + 'static,
{
    fn allocate(&mut self, t: &ObjectId, pointer: &P) {
        let (pages, _) = &self.allocator.page_mapping[*pointer];
        let vn = self.aux.obj_info().typ(*t).with_normalized_p();
        let rv = ResidentalValue::new(Value::new(*t, self.machine.device(), vn), *pointer);

        let vs = pages.len() * (self.allocator.page_size as usize);

        self.machine.allocate(
            AllocMethod::Paged {
                va_size: vs,
                pa: pages.iter().map(|i| usize::from(*i)).collect(),
            },
            rv,
        );
    }

    fn deallocate(&mut self, t: &ObjectId, pointer: &P) {
        self.machine
            .deallocate_object(*t, pointer, self.aux.obj_info(), AllocVariant::Paged);
    }

    fn transfer(
        &mut self,
        t: &ObjectId,
        from_pointer: &P,
        to_device: Device,
        to_pointer: &P,
    ) -> RealizationResponse<'s, ObjectId, P, Result<(), Error<'s>>, Rt> {
        Response::Continue(Continuation::transfer_object(
            self.machine.device(),
            *from_pointer,
            to_device,
            *to_pointer,
            *t,
        ))
    }
}

impl<'s, P: UsizeId + 'static, Rt: RuntimeType, D: DeviceMarker> Allocator<'s, ObjectId, P, Rt>
    for PageAllocator<'s, P, Rt, D>
{
    fn handle<'a, 'b, 'c, 'd, 'i>(
        &'a mut self,
        machine: planning::MachineHandle<'b, 's, ObjectId, P>,
        aux: &'c mut AuxiliaryInfo<'i, Rt>,
    ) -> Box<dyn AllocatorHandle<'s, ObjectId, P, Rt> + 'd>
    where
        'a: 'd,
        'b: 'd,
        'c: 'd,
        'i: 'd,
    {
        Box::new(Handle {
            allocator: self,
            machine,
            aux,
        })
    }

    fn realizer<'a, 'b, 'c, 'd, 'i>(
        &'a mut self,
        machine: realization::MachineHandle<'b, 's, P, Rt>,
        aux: &'c mut AuxiliaryInfo<'i, Rt>,
    ) -> Box<dyn AllocatorRealizer<'s, ObjectId, P, Rt> + 'd>
    where
        'a: 'd,
        'b: 'd,
        'c: 'd,
        'i: 'd,
    {
        Box::new(Realizer {
            allocator: self,
            machine,
            aux,
        })
    }

    fn allcate_pointer(&mut self) -> P {
        self.page_mapping
            .push((vec![], Size::Integral(IntegralSize(0))))
    }

    fn inner<'t>(&'t mut self) -> Option<&'t mut dyn Allocator<'s, ObjectId, P, Rt>> {
        None
    }
}
