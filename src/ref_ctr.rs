use std::{ops::Deref, rc::Rc};

#[derive(Debug)]
pub struct RefCtr<T> {
    referred: Rc<T>,
}
impl<T> RefCtr<T> {
    #[allow(unused)]
    pub fn new(value: T) -> Self {
        let referred = Rc::new(value);
        Self { referred }
    }
}
impl<T> Clone for RefCtr<T> {
    fn clone(&self) -> Self {
        Self {
            referred: Rc::clone(&self.referred),
        }
    }
}
impl<T> Deref for RefCtr<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        &self.referred
    }
}
impl<T> AsRef<T> for RefCtr<T> {
    fn as_ref(&self) -> &T {
        &self.referred
    }
}
