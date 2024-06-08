use std::{
    num::NonZeroUsize,
    sync::{Arc, Mutex},
};

use crate::{
    node::Node,
    tensor::{IndexIter, OwnedShape, Tensor},
};

pub fn filter_layer(
    inputs: Tensor<'_, Arc<Mutex<Node>>>,
    stride: NonZeroUsize,
    filter_shape: &[usize],
    mut create_filter: impl FnMut(FilterParams) -> Arc<Mutex<Node>>,
) -> (Vec<Arc<Mutex<Node>>>, OwnedShape) {
    let mut shape = inputs.shape().to_vec();
    shape
        .iter_mut()
        .zip(filter_shape.iter().copied())
        .for_each(|(x, kernel_size)| *x = x.saturating_sub(kernel_size));
    let start_range = shape.iter().copied().map(|x| 0..x).collect::<Vec<_>>();
    let mut start_indices = IndexIter::new(&start_range, stride);
    let mut filters = vec![];
    while let Some(start_index) = start_indices.next_index() {
        let range = start_index
            .iter()
            .copied()
            .zip(filter_shape.iter().copied())
            .map(|(start, len)| start..(start + len))
            .collect::<Vec<_>>();
        let mut filter_input_indices = IndexIter::new(&range, NonZeroUsize::new(1).unwrap());
        let mut filter_inputs = vec![];
        while let Some(kernel_input_index) = filter_input_indices.next_index() {
            let node = inputs.get(kernel_input_index).unwrap();
            filter_inputs.push(Arc::clone(node));
        }
        let params = FilterParams {
            i: filters.len(),
            inputs: filter_inputs,
        };
        let filter = create_filter(params);
        filters.push(filter);
    }
    (filters, start_indices.shape())
}

#[derive(Debug, Clone)]
pub struct FilterParams {
    pub i: usize,
    pub inputs: Vec<Arc<Mutex<Node>>>,
}
