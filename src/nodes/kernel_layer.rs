use std::{
    num::NonZeroUsize,
    sync::{Arc, Mutex},
};

use crate::{
    node::Node,
    param::ParamInjector,
    tensor::{OwnedShape, Tensor},
};

use super::{
    filter_layer::{filter_layer, FilterParams},
    linear_node::{self, linear_node},
};

pub struct KernelConfig<'a> {
    pub shape: &'a [usize],
    pub initial_weights: Option<Box<dyn Fn() -> Vec<f64>>>,
    pub initial_bias: Option<Box<dyn Fn() -> f64>>,
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
) -> (Vec<Arc<Mutex<Node>>>, OwnedShape) {
    let create_filter = |params: FilterParams| -> Arc<Mutex<Node>> {
        let weights = kernel.initial_weights.as_ref().map(|f| f());
        let bias = kernel.initial_bias.as_ref().map(|f| f());
        let param_injection = param_injection.as_mut().map(|x| {
            let weight_name = format!("{}:kernel.{}:weights", x.name, params.i);
            let bias_name = format!("{}:kernel.{}:bias", x.name, params.i);
            linear_node::ParamInjection {
                injector: x.injector,
                weight_name,
                bias_name,
            }
        });
        let feature_node =
            linear_node(params.inputs, weights, bias, kernel.lambda, param_injection);
        feature_node.unwrap()
    };
    filter_layer(inputs, stride, kernel.shape, create_filter)
}
