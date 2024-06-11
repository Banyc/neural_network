#![allow(unused)]

use std::{
    cell::{RefCell, UnsafeCell},
    ops::{Deref, DerefMut},
};

#[derive(Debug)]
pub struct MutCell<T> {
    #[cfg(debug_assertions)]
    cell: RefCell<T>,
    #[cfg(not(debug_assertions))]
    cell: UnsafeCell<T>,
}
impl<T> MutCell<T> {
    pub fn new(value: T) -> Self {
        Self {
            #[cfg(debug_assertions)]
            cell: RefCell::new(value),
            #[cfg(not(debug_assertions))]
            cell: UnsafeCell::new(value),
        }
    }

    #[cfg(debug_assertions)]
    pub fn borrow_mut(&self) -> impl DerefMut<Target = T> + '_ {
        self.cell.borrow_mut()
    }
    #[cfg(not(debug_assertions))]
    pub fn borrow_mut(&self) -> impl DerefMut<Target = T> + '_ {
        let value = unsafe { &mut *self.cell.get() };
        SafeWrapMut::new(value)
    }

    #[cfg(debug_assertions)]
    pub fn borrow(&self) -> impl Deref<Target = T> + '_ {
        self.cell.borrow()
    }
    #[cfg(not(debug_assertions))]
    pub fn borrow(&self) -> impl Deref<Target = T> + '_ {
        let value = unsafe { &*self.cell.get() };
        SafeWrap::new(value)
    }
}

struct SafeWrap<'a, T> {
    value: &'a T,
}
impl<'a, T> SafeWrap<'a, T> {
    pub fn new(value: &'a T) -> Self {
        Self { value }
    }
}
impl<T> Deref for SafeWrap<'_, T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        self.value
    }
}

struct SafeWrapMut<'a, T> {
    value: &'a mut T,
}
impl<'a, T> SafeWrapMut<'a, T> {
    pub fn new(value: &'a mut T) -> Self {
        Self { value }
    }
}
impl<T> Deref for SafeWrapMut<'_, T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        self.value
    }
}
impl<T> DerefMut for SafeWrapMut<'_, T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.value
    }
}
