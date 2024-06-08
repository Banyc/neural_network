use std::sync::{Arc, Mutex};

use crate::{
    node::Node,
    param::ParamInjector,
    tensor::{NonZeroShape, OwnedShape, Stride, Tensor},
};

use super::{
    kernel_layer::{kernel_layer, KernelParams},
    linear_node::{self, linear_node},
};

pub struct KernelConfig<'a> {
    pub shape: &'a NonZeroShape,
    pub initial_weights: Option<Box<dyn Fn() -> Vec<f64>>>,
    pub initial_bias: Option<Box<dyn Fn() -> f64>>,
    pub lambda: Option<f64>,
}

#[derive(Debug)]
pub struct ParamInjection<'a> {
    pub injector: &'a mut ParamInjector,
    pub name: String,
}

pub fn conv_layer(
    inputs: Tensor<'_, Arc<Mutex<Node>>>,
    stride: &Stride,
    kernel: KernelConfig,
    mut param_injection: Option<ParamInjection<'_>>,
) -> (Vec<Arc<Mutex<Node>>>, OwnedShape) {
    let create_filter = |params: KernelParams| -> Arc<Mutex<Node>> {
        let weights = kernel.initial_weights.as_ref().map(|f| f());
        let bias = kernel.initial_bias.as_ref().map(|f| f());
        let param_injection = param_injection.as_mut().map(|x| {
            let weights_name = format!("{}:kernel.{}:weights", x.name, params.i);
            let bias_name = format!("{}:kernel.{}:bias", x.name, params.i);
            linear_node::ParamInjection {
                injector: x.injector,
                weights_name,
                bias_name,
            }
        });
        let feature_node =
            linear_node(params.inputs, weights, bias, kernel.lambda, param_injection);
        feature_node.unwrap()
    };
    kernel_layer(inputs, stride, kernel.shape, create_filter)
}
