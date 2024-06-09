use std::num::NonZeroUsize;

#[derive(Debug, PartialEq, Eq)]
pub struct Tensor<'a, T> {
    data: &'a [T],
    shape: &'a Shape,
}
impl<'a, T> Tensor<'a, T> {
    pub fn new(data: &'a [T], shape: &'a Shape) -> Option<Self> {
        let n = shape.iter().copied().product::<usize>();
        if data.len() != n {
            return None;
        }
        Some(Self { data, shape })
    }
}
impl<T> Tensor<'_, T> {
    pub fn get(&self, index: &Index) -> Option<&T> {
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

pub fn append_tensors<T>(tensor_data: Vec<Vec<T>>, shape: &Shape) -> (Vec<T>, OwnedNonZeroShape) {
    let new_dim = NonZeroUsize::new(tensor_data.len()).unwrap();
    let mut appended_tensor_data = vec![];
    for tensor_data in tensor_data {
        Tensor::new(&tensor_data, shape).unwrap();
        appended_tensor_data.extend(tensor_data);
    }
    let shape = canonical_shape(shape).unwrap();
    let mut appended_shape = shape;
    appended_shape.push(new_dim);
    (appended_tensor_data, appended_shape)
}

pub type Index = [usize];
pub type OwnedIndex = Vec<usize>;

pub type Shape = [usize];
pub type OwnedShape = Vec<usize>;
pub type NonZeroShape = [NonZeroUsize];
pub type OwnedNonZeroShape = Vec<NonZeroUsize>;
/// Remove dimensions with length 1 and collapse empty shapes
pub fn canonical_shape(shape: &Shape) -> Option<OwnedNonZeroShape> {
    let mut canonical = vec![];
    for x in shape.iter().copied() {
        if x == 1 {
            continue;
        }
        let x = NonZeroUsize::new(x)?;
        canonical.push(x);
    }
    Some(canonical)
}
pub fn non_zero_to_shape(shape: &NonZeroShape) -> OwnedShape {
    shape.iter().map(|x| x.get()).collect()
}
pub fn shape_to_non_zero(shape: &Shape) -> Option<OwnedNonZeroShape> {
    shape.iter().copied().map(NonZeroUsize::new).collect()
}

pub type Range = [std::ops::Range<usize>];
pub type OwnedRange = Vec<std::ops::Range<usize>>;

pub type Stride = [NonZeroUsize];
pub type OwnedStride = Vec<NonZeroUsize>;
pub fn primitive_to_stride(stride: &[usize]) -> Option<OwnedStride> {
    stride.iter().copied().map(NonZeroUsize::new).collect()
}

#[derive(Debug, Clone)]
pub struct IndexIter<'a> {
    range: &'a Range,
    stride: &'a Stride,
    index: Option<OwnedIndex>,
    done: bool,
}
impl<'a> IndexIter<'a> {
    pub fn new(range: &'a Range, stride: &'a Stride) -> Self {
        assert_eq!(range.len(), stride.len());
        // let index = range.iter().map(|x| x.start).collect();
        Self {
            range,
            stride,
            index: None,
            done: false,
        }
    }
}
impl IndexIter<'_> {
    pub fn next_index(&mut self) -> Option<&OwnedIndex> {
        if self.done {
            return None;
        }
        let index = if self.index.is_none() {
            let out_of_end = self.range.iter().any(|x| x.end <= x.start);
            if out_of_end {
                None
            } else {
                Some(self.range.iter().map(|x| x.start).collect::<OwnedIndex>())
            }
        } else {
            next_index(self.index.take().unwrap(), self.range, self.stride)
        };
        let Some(index) = index else {
            self.done = true;
            return None;
        };
        self.index = Some(index);
        return Some(self.index.as_ref().unwrap());

        fn next_index(mut index: OwnedIndex, range: &Range, stride: &Stride) -> Option<OwnedIndex> {
            assert_eq!(index.len(), range.len());
            assert_eq!(index.len(), stride.len());
            for dim in 0..index.len() {
                if index[dim] + stride[dim].get() < range[dim].end {
                    index[dim] += stride[dim].get();
                    return Some(index);
                }
                index[dim] = range[dim].start;
            }
            None
        }
    }

    pub fn shape(&self) -> OwnedShape {
        self.range
            .iter()
            .zip(self.stride.iter())
            .map(|(range, stride)| range.len().div_ceil(stride.get()))
            .collect()
    }
}

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

    #[test]
    fn test_iter_len() {
        assert_eq!((0..2).len(), 2);
    }
}
