use std::num::NonZeroUsize;

use graph::NodeIdx;

use crate::tensor::{IndexIter, NonZeroShape, OwnedShape, OwnedStride, Shape, Stride, Tensor};

#[derive(Debug, Clone)]
pub struct KernelLayerConfig<'a> {
    pub stride: &'a Stride,
    pub kernel_shape: &'a NonZeroShape,
    pub assert_output_shape: Option<&'a Shape>,
}
pub fn kernel_layer(
    inputs: Tensor<'_, NodeIdx>,
    config: KernelLayerConfig<'_>,
    mut create_kernel: impl FnMut(KernelParams) -> NodeIdx,
) -> (Vec<NodeIdx>, OwnedShape) {
    let mut shape = inputs.shape().to_vec();
    shape
        .iter_mut()
        .zip(config.kernel_shape.iter().copied())
        .for_each(|(x, kernel_size)| *x = x.saturating_sub(kernel_size.get() - 1));
    let start_range = shape.iter().copied().map(|x| 0..x).collect::<Vec<_>>();
    let mut start_indices = IndexIter::new(&start_range, config.stride);
    let mut kernels = vec![];
    while let Some(start_index) = start_indices.next_index() {
        let range = start_index
            .iter()
            .copied()
            .zip(config.kernel_shape.iter().copied())
            .map(|(start, len)| start..(start + len.get()))
            .collect::<Vec<_>>();
        let stride = (0..range.len())
            .map(|_| NonZeroUsize::new(1).unwrap())
            .collect::<OwnedStride>();
        let mut kernel_input_indices = IndexIter::new(&range, &stride);
        let mut kernel_inputs = vec![];
        while let Some(kernel_input_index) = kernel_input_indices.next_index() {
            let node = *inputs.get(kernel_input_index).unwrap();
            kernel_inputs.push(node);
        }
        let params = KernelParams {
            i: kernels.len(),
            inputs: kernel_inputs,
        };
        let kernel = create_kernel(params);
        kernels.push(kernel);
    }
    if let Some(assert_output_shape) = config.assert_output_shape {
        assert_eq!(assert_output_shape, start_indices.shape());
    }
    (kernels, start_indices.shape())
}

#[derive(Debug, Clone)]
pub struct KernelParams {
    pub i: usize,
    pub inputs: Vec<NodeIdx>,
}
