use std::{
    num::NonZeroUsize,
    sync::{Arc, Mutex},
};

use crate::{
    node::Node,
    nodes::weight_node,
    param::ParamInjector,
    tensor::{IndexIter, Tensor},
};

pub struct KernelConfig<'a> {
    pub shape: &'a [usize],
    pub initial_weights: Option<Box<dyn Fn() -> Vec<f64>>>,
    pub lambda: Option<f64>,
}

#[derive(Debug)]
pub struct ParamInjection<'a> {
    pub injector: &'a mut ParamInjector,
    pub name: String,
}

pub fn kernel_layer(
    inputs: Tensor<'_, Arc<Mutex<Node>>>,
    stride: NonZeroUsize,
    kernel: KernelConfig,
    mut param_injection: Option<ParamInjection<'_>>,
) -> Vec<Arc<Mutex<Node>>> {
    let mut shape = inputs.shape().to_vec();
    shape
        .iter_mut()
        .zip(kernel.shape.iter().copied())
        .for_each(|(x, kernel_size)| *x = x.saturating_sub(kernel_size));
    let start_range = shape.into_iter().map(|x| 0..x).collect::<Vec<_>>();
    let mut start_indices = IndexIter::new(&start_range, stride);
    let mut kernels = vec![];
    while let Some(start_index) = start_indices.next_index() {
        let range = start_index
            .iter()
            .copied()
            .zip(kernel.shape.iter().copied())
            .map(|(start, len)| start..(start + len))
            .collect::<Vec<_>>();
        let mut kernel_input_indices = IndexIter::new(&range, NonZeroUsize::new(1).unwrap());
        let mut kernel_inputs = vec![];
        while let Some(kernel_input_index) = kernel_input_indices.next_index() {
            let node = inputs.get(kernel_input_index).unwrap();
            kernel_inputs.push(Arc::clone(node));
        }
        let weights = kernel.initial_weights.as_ref().map(|f| f());
        let feature_node = weight_node::weight_node(kernel_inputs, weights, kernel.lambda).unwrap();
        let feature_node = Arc::new(Mutex::new(feature_node));
        if let Some(param_injection) = &mut param_injection {
            let kernel_i = kernels.len();
            let name = format!("{}:kernel.{}", param_injection.name, kernel_i);
            param_injection
                .injector
                .insert_node(name, Arc::clone(&feature_node));
        }
        kernels.push(feature_node);
    }
    kernels
}
