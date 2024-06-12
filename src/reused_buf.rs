#[derive(Debug)]
pub struct ReusedBuffers<T> {
    buffers: Vec<Vec<T>>,
    max: usize,
}
impl<T> ReusedBuffers<T> {
    pub fn new(max: usize) -> Self {
        Self {
            buffers: vec![],
            max,
        }
    }

    pub fn take(&mut self) -> Vec<T> {
        self.buffers.pop().unwrap_or_default()
    }

    pub fn put(&mut self, mut buf: Vec<T>) {
        if self.buffers.len() == self.max {
            return;
        }
        buf.clear();
        self.buffers.push(buf);
    }
}

#[derive(Debug)]
pub struct ReusedRefBuffers<T: 'static> {
    buf: ReusedBuffers<&'static T>,
}
#[allow(unused)]
impl<T: 'static> ReusedRefBuffers<T> {
    pub fn new(max: usize) -> Self {
        Self {
            buf: ReusedBuffers::new(max),
        }
    }

    pub fn take(&mut self) -> Vec<&T> {
        self.buf.take()
    }

    pub fn put(&mut self, buf: Vec<&T>) {
        let buf = into_static(buf);
        self.buf.put(buf);
    }
}

fn into_static<T>(mut buf: Vec<&T>) -> Vec<&'static T> {
    buf.clear();
    buf.into_iter()
        .map(|_| -> &'static T { unreachable!() })
        .collect()
}
