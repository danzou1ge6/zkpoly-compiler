use std::{
    alloc::{alloc, dealloc, Layout},
    any::Any,
};

#[derive(Debug, Clone)]
pub struct AnyWrapper {
    /// Pointer to the value
    pub value: *mut *mut dyn Any,
}

unsafe impl Send for AnyWrapper {}
unsafe impl Sync for AnyWrapper {}

impl AnyWrapper {
    /// Create a new `AnyWrapper` from a pointer to a value
    pub fn new(payload: Box<dyn Any>) -> Self {
        // get the payload into a pointer
        let ptr = Box::into_raw(payload);
        // allocate the pointer to the pointer
        let value = unsafe { alloc(Layout::new::<*mut dyn Any>()) as *mut *mut dyn Any };
        // write the pointer to the pointer
        unsafe {
            // write the pointer to the pointer
            *value = ptr;
        }

        Self { value }
    }

    // note that once this function is called, all registers cloned from this struct
    // will be invalidated, because the payload is deallocated
    // and the pointer to the payload is deallocated
    pub fn dealloc(&mut self) {
        unsafe {
            // deallocate the payload
            let _ = Box::from_raw(*self.value);
            // deallocate the pointer to the pointer
            dealloc(self.value as *mut u8, Layout::new::<*mut dyn Any>());
        }
    }

    /// Get the value as a reference to `T`
    pub fn get(&self) -> &dyn Any {
        unsafe {
            // fist we need get the pointer to the payload
            let payload = *self.value;
            // then we need to dereference it to get the value
            return &*payload;
        }
    }

    pub fn replace(&mut self, payload: Box<dyn Any + Send + Sync>) {
        unsafe {
            // deallocate the payload
            let _ = Box::from_raw(*self.value);
            // get the pointer to the payload
            let ptr = Box::into_raw(payload);
            // point to this new payload ptr
            *self.value = ptr
        }
    }
}

#[test]
fn test_any_wrapper() {
    let mut any = AnyWrapper::new(Box::new(1));
    assert_eq!(any.get().downcast_ref::<i32>(), Some(&1));
    any.replace(Box::new(2));
    assert_eq!(any.get().downcast_ref::<i32>(), Some(&2));
    let any2 = any.clone();
    assert_eq!(any2.get().downcast_ref::<i32>(), Some(&2));
    any.replace(Box::new(3));
    // the payload is not deallocated here, because it is not owned by this struct
    assert_eq!(any2.get().downcast_ref::<i32>(), Some(&3));
    // deallocate the payload
    any.dealloc();
}
