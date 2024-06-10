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
