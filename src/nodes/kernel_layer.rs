use std::{
    num::NonZeroUsize,
    sync::{Arc, Mutex},
};

use crate::{node::Node, nodes::weight_node, param::ParamInjector, tensor::Tensor};

use super::filter_layer::{filter_layer, FilterParams};

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
    let create_filter = |params: FilterParams| -> Arc<Mutex<Node>> {
        let weights = kernel.initial_weights.as_ref().map(|f| f());
        let feature_node = weight_node::weight_node(params.inputs, weights, kernel.lambda).unwrap();
        let feature_node = Arc::new(Mutex::new(feature_node));
        if let Some(param_injection) = &mut param_injection {
            let name = format!("{}:kernel.{}", param_injection.name, params.i);
            param_injection
                .injector
                .insert_node(name, Arc::clone(&feature_node));
        }
        feature_node
    };
    filter_layer(inputs, stride, kernel.shape, create_filter)
}
