use std::num::NonZeroUsize;

#[derive(Debug, PartialEq, Eq)]
pub struct Tensor<'a, T> {
    data: &'a [T],
    shape: &'a [usize],
}
impl<'a, T> Tensor<'a, T> {
    pub fn new(data: &'a [T], shape: &'a [usize]) -> Option<Self> {
        let n = shape.iter().copied().product::<usize>();
        if data.len() != n {
            return None;
        }
        Some(Self { data, shape })
    }
}
impl<T> Tensor<'_, T> {
    pub fn get(&self, index: &[usize]) -> Option<&T> {
        if self.shape.len() != index.len() {
            return None;
        }
        if self
            .shape
            .iter()
            .copied()
            .zip(index.iter().copied())
            .any(|(shape, index)| shape <= index)
        {
            return None;
        }
        let mut pos = 0;
        let mut mag = 1;
        for (i, len) in index.iter().copied().zip(self.shape.iter().copied()) {
            pos += i * mag;
            mag *= len;
        }
        self.data.get(pos)
    }

    pub fn shape(&self) -> &[usize] {
        self.shape
    }
}
impl<T> Clone for Tensor<'_, T> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<T> Copy for Tensor<'_, T> {}

pub struct IndexIter<'a> {
    range: &'a Range,
    stride: NonZeroUsize,
    index: Option<Vec<usize>>,
}
impl<'a> IndexIter<'a> {
    pub fn new(range: &'a Range, stride: NonZeroUsize) -> Self {
        let index = range.iter().map(|x| x.start).collect();
        Self {
            range,
            stride,
            index: Some(index),
        }
    }
}
impl IndexIter<'_> {
    pub fn next_index(&mut self) -> Option<&Vec<usize>> {
        let index = next_index(self.index.take().unwrap(), self.range, self.stride)?;
        self.index = Some(index);
        return Some(self.index.as_ref().unwrap());

        fn next_index(
            mut index: Vec<usize>,
            range: &Range,
            stride: NonZeroUsize,
        ) -> Option<Vec<usize>> {
            assert_eq!(index.len(), range.len());
            for dim in 0..index.len() {
                if index[dim] + stride.get() < range[dim].end {
                    index[dim] += stride.get();
                    return Some(index);
                }
                index[dim] = range[dim].start;
            }
            None
        }
    }
}

pub type Range = [std::ops::Range<usize>];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_indexing() {
        let data = [
            //
            0, 1, 2, 3, //
            4, 5, 6, 7, //
            8, 9, 10, 11, //
            //
            12, 13, 14, 15, //
            16, 17, 18, 19, //
            20, 21, 22, 23, //
        ];
        let shape = [4, 3, 2];
        let tensor = Tensor::new(&data, &shape).unwrap();
        assert_eq!(*tensor.get(&[0, 0, 0]).unwrap(), 0);
        assert_eq!(*tensor.get(&[1, 0, 0]).unwrap(), 1);
        assert_eq!(*tensor.get(&[2, 0, 0]).unwrap(), 2);
        assert_eq!(*tensor.get(&[3, 0, 0]).unwrap(), 3);
        assert_eq!(*tensor.get(&[0, 1, 0]).unwrap(), 4);
        assert_eq!(*tensor.get(&[1, 1, 0]).unwrap(), 5);
        assert_eq!(*tensor.get(&[0, 2, 0]).unwrap(), 8);
        assert_eq!(*tensor.get(&[1, 2, 0]).unwrap(), 9);
        assert_eq!(*tensor.get(&[0, 0, 1]).unwrap(), 12);
        assert_eq!(*tensor.get(&[1, 0, 1]).unwrap(), 13);
    }
}
