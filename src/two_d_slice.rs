#[derive(Debug, Clone)]
pub struct TwoDSlice<'a, T> {
    slice: &'a [T],
    chunk_size: usize,
}
impl<'a, T> TwoDSlice<'a, T> {
    pub fn new(slice: &'a [T], chunk_size: usize) -> Self {
        if slice.is_empty() {
            assert!(chunk_size == 0);
        } else {
            assert!(chunk_size != 0);
            assert!(slice.len() % chunk_size == 0);
        }
        Self { slice, chunk_size }
    }

    pub fn slice(self, chunk_index: usize) -> &'a [T] {
        if self.slice.is_empty() {
            return self.slice;
        }
        let start = self.chunk_size * chunk_index;
        let end = start + self.chunk_size;
        &self.slice[start..end]
    }
}
