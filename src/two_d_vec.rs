#[derive(Debug, Clone)]
pub struct TwoDVec<T> {
    vec: Vec<T>,
    chunk_size: usize,
}
impl<T> TwoDVec<T> {
    pub fn new(vec: Vec<T>, chunk_size: usize) -> Self {
        if vec.is_empty() {
            assert!(chunk_size == 0);
        } else {
            assert!(chunk_size != 0);
            assert!(vec.len() % chunk_size == 0);
        }
        Self { vec, chunk_size }
    }

    pub fn slice(&self, chunk_index: usize) -> &[T] {
        if self.vec.is_empty() {
            return &self.vec;
        }
        let start = self.chunk_size * chunk_index;
        let end = start + self.chunk_size;
        &self.vec[start..end]
    }

    pub fn into_vec(self) -> Vec<T> {
        self.vec
    }
}
