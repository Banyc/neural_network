use std::{
    num::NonZeroUsize,
    sync::{Arc, Mutex},
};

use crate::{
    node::Node,
    tensor::{IndexIter, OwnedShape, Stride, Tensor},
};

pub fn kernel_layer(
    inputs: Tensor<'_, Arc<Mutex<Node>>>,
    stride: &Stride,
    kernel_shape: &[usize],
    mut create_kernel: impl FnMut(KernelParams) -> Arc<Mutex<Node>>,
) -> (Vec<Arc<Mutex<Node>>>, OwnedShape) {
    let mut shape = inputs.shape().to_vec();
    shape
        .iter_mut()
        .zip(kernel_shape.iter().copied())
        .for_each(|(x, kernel_size)| *x = x.saturating_sub(kernel_size));
    let start_range = shape.iter().copied().map(|x| 0..x).collect::<Vec<_>>();
    let mut start_indices = IndexIter::new(&start_range, stride);
    let mut kernels = vec![];
    while let Some(start_index) = start_indices.next_index() {
        let range = start_index
            .iter()
            .copied()
            .zip(kernel_shape.iter().copied())
            .map(|(start, len)| start..(start + len))
            .collect::<Vec<_>>();
        let stride = (0..range.len())
            .map(|_| NonZeroUsize::new(1).unwrap())
            .collect::<Vec<NonZeroUsize>>();
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
    pub inputs: Vec<Arc<Mutex<Node>>>,
}
