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
        pub fn decide_victim(&mut self) -> Option<PageId> {
            let index = self
                .next_use
                .query_max_info(0, self.occupied.len() - 1)
                .unwrap()
                .index;
            let index = PageId::from(index);

            if self.occupied[index] {
                Some(index)
            } else {
                None
            }
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
    }
}

use pages::{PageId, Pages};

pub struct PageAllocator<P> {
    page_size: u64,
    pages: Pages,
    living_objects: BTreeMap<ObjectId, (P, Size)>,
    page_mapping: Heap<P, Vec<PageId>>,
    page_objects: Heap<PageId, Option<ObjectId>>,
}

impl<P> PageAllocator<P> {
    pub fn new(number_pages: usize, page_size: u64) -> Self {
        Self {
            page_size,
            pages: Pages::new(number_pages),
            living_objects: BTreeMap::new(),
            page_mapping: Heap::new(),
            page_objects: Heap::repeat(None, number_pages),
        }
    }
}

pub struct Handle<'a, 'm, 's, 'au, 'i, P, Rt: RuntimeType> {
    allocator: &'a mut PageAllocator<P>,
    machine: planning::MachineHandle<'m, 's, ObjectId, P>,
    aux: &'au mut AuxiliaryInfo<'i, Rt>,
}

impl<'a, 'm, 's, 'au, 'i, P, Rt: RuntimeType> Handle<'a, 'm, 's, 'au, 'i, P, Rt>
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

        let p = self.allocator.page_mapping.push(pages);
        self.allocator.living_objects.insert(object, (p, size));

        p
    }

    fn deallocate_pages(&mut self, victim_object: ObjectId) -> (P, Size) {
        let (victim_pointer, victim_size) = self
            .allocator
            .living_objects
            .remove(&victim_object)
            .unwrap_or_else(|| panic!("object {:?} not allocated", victim_object));
        let victim_pages = self.allocator.page_mapping[victim_pointer].clone();

        victim_pages.iter().for_each(|page| {
            self.allocator.pages.free(*page);
            self.allocator.page_objects[*page] = None;
        });

        (victim_pointer, victim_size)
    }
}

impl<'a, 'm, 's, 'au, 'i, P, Rt: RuntimeType> AllocatorHandle<'s, ObjectId, P, Rt>
    for Handle<'a, 'm, 's, 'au, 'i, P, Rt>
where
    P: UsizeId + 'static,
{
    fn access(&mut self, t: &ObjectId) -> Option<P> {
        self.allocator
            .living_objects
            .get(t)
            .map(|(a, _)| a)
            .cloned()
    }

    fn allocate<'f>(
        &mut self,
        size: Size,
        t: &ObjectId,
    ) -> Response<'f, planning::Machine<'s, ObjectId, P>, ObjectId, P, Result<P, Error<'s>>, Rt>
    where
        's: 'f,
    {
        if let Some(_) = self.allocator.living_objects.get(t) {
            panic!("{:?} already allocated", t);
        }

        let number = u64::from(size).div_ceil(self.allocator.page_size) as usize;
        let next_use = self.aux.query_next_use(t.clone(), self.device());

        if number > self.allocator.pages.number() {
            return Response::Complete(Err(Error::VertexInputsOutputsNotAccommodated(None)));
        }

        if let Some(pages) = self.allocator.pages.try_find_free_pages(number) {
            let p = self.allocate_pages(pages, *t, next_use, size);
            self.machine.allocate(*t, p);
            Response::Complete(Ok(p))
        } else {
            let victim_page = self
                .allocator
                .pages
                .decide_victim()
                .expect("since there are occupied pages, a victim page must can be found");
            let victim_object = self.allocator.page_objects[victim_page]
                .expect("an occupied page must belong to some object");
            let (victim_pointer, victim_size) = self.deallocate_pages(victim_object);

            let t = *t;
            let this_device = self.device();

            Response::Continue(
                Continuation::simple_eject(
                    self.device(),
                    victim_pointer,
                    victim_object,
                    victim_size,
                )
                .bind_result(move |_| {
                    Continuation::new(move |allocators, machine, aux| {
                        allocators
                            .handle(this_device, machine, aux)
                            .allocate(size, &t)
                    })
                }),
            )
        }
    }
    fn claim<'f>(
        &mut self,
        t: &ObjectId,
        size: Size,
        from: Device,
    ) -> Response<'f, planning::Machine<'s, ObjectId, P>, ObjectId, P, Result<(), Error<'s>>, Rt>
    where
        's: 'f,
    {
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

    fn deallocate<'f>(
        &mut self,
        t: &ObjectId,
    ) -> Response<'f, planning::Machine<'s, ObjectId, P>, ObjectId, P, (), Rt>
    where
        's: 'f,
    {
        let (victim_pointer, _) = self.deallocate_pages(*t);
        self.machine.deallocate(*t, victim_pointer);
        Response::Complete(())
    }

    fn device(&self) -> Device {
        self.machine.device()
    }

    fn reuse(&mut self, new_object: ObjectId, old_object: ObjectId) {
        if let Some((p, size)) = self.allocator.living_objects.remove(&old_object) {
            self.allocator.living_objects.insert(new_object, (p, size));
        } else {
            panic!("object {:?} not allocated", old_object)
        }
    }
}
