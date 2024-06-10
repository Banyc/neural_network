use std::{num::NonZeroUsize, sync::Arc};

use crate::{
    node::SharedNode,
    tensor::{IndexIter, NonZeroShape, OwnedShape, OwnedStride, Stride, Tensor},
};

pub fn kernel_layer(
    inputs: Tensor<'_, SharedNode>,
    stride: &Stride,
    kernel_shape: &NonZeroShape,
    mut create_kernel: impl FnMut(KernelParams) -> SharedNode,
) -> (Vec<SharedNode>, OwnedShape) {
    let mut shape = inputs.shape().to_vec();
    shape
        .iter_mut()
        .zip(kernel_shape.iter().copied())
        .for_each(|(x, kernel_size)| *x = x.saturating_sub(kernel_size.get() - 1));
    let start_range = shape.iter().copied().map(|x| 0..x).collect::<Vec<_>>();
    let mut start_indices = IndexIter::new(&start_range, stride);
    let mut kernels = vec![];
    while let Some(start_index) = start_indices.next_index() {
        let range = start_index
            .iter()
            .copied()
            .zip(kernel_shape.iter().copied())
            .map(|(start, len)| start..(start + len.get()))
            .collect::<Vec<_>>();
        let stride = (0..range.len())
            .map(|_| NonZeroUsize::new(1).unwrap())
            .collect::<OwnedStride>();
        let mut kernel_input_indices = IndexIter::new(&range, &stride);
        let mut kernel_inputs = vec![];
        while let Some(kernel_input_index) = kernel_input_indices.next_index() {
            let node = inputs.get(kernel_input_index).unwrap();
            kernel_inputs.push(Arc::clone(node));
        }
        let params = KernelParams {
            i: kernels.len(),
            inputs: kernel_inputs,
        };
        let kernel = create_kernel(params);
        kernels.push(kernel);
    }
    (kernels, start_indices.shape())
}

#[derive(Debug, Clone)]
pub struct KernelParams {
    pub i: usize,
    pub inputs: Vec<SharedNode>,
}
