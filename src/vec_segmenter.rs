#[derive(Debug, Clone)]
pub struct VecSegmenter<T, const N: usize> {
    buf: Vec<T>,
    segment_len: [usize; N],
}
impl<T, const N: usize> VecSegmenter<T, N> {
    pub fn new(buf: Vec<T>, segment_len: [usize; N]) -> Self {
        assert!(*segment_len.last().unwrap() <= buf.len());
        Self { buf, segment_len }
    }

    fn range(&self, index: usize) -> Option<std::ops::Range<usize>> {
        let len = *self.segment_len.get(index)?;
        let mut start = 0;
        for len in self.segment_len.iter().take(index) {
            start += len;
        }
        let end = start + len;
        Some(start..end)
    }

    pub fn segment(&self, index: usize) -> Option<&[T]> {
        let range = self.range(index)?;
        Some(&self.buf[range])
    }
    pub fn segment_mut(&mut self, index: usize) -> Option<&mut [T]> {
        let range = self.range(index)?;
        Some(&mut self.buf[range])
    }

    pub fn into_vec(self) -> Vec<T> {
        self.buf
    }
}
