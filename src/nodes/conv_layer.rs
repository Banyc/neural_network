use std::{
    num::NonZeroUsize,
    sync::{Arc, Mutex},
};

use crate::{
    node::Node,
    param::ParamInjection,
    tensor::{NonZeroShape, OwnedShape, Stride, Tensor},
};

use super::{
    kernel_layer::{kernel_layer, KernelParams},
    linear_node::linear_node,
};

#[derive(Clone)]
pub struct KernelConfig<'a> {
    pub shape: &'a NonZeroShape,
    pub initial_weights: Option<&'a dyn Fn() -> Vec<f64>>,
    pub initial_bias: Option<&'a dyn Fn() -> f64>,
    pub lambda: Option<f64>,
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

        let param_injection = param_injection
            .as_mut()
            .map(|x| x.name_append(&format!(":kernel.{}", params.i)));
        let feature_node =
            linear_node(params.inputs, weights, bias, kernel.lambda, param_injection);
        feature_node.unwrap()
    };
    kernel_layer(inputs, stride, kernel.shape, create_filter)
}

pub fn deep_conv_layer(
    inputs: Tensor<'_, Arc<Mutex<Node>>>,
    stride: &Stride,
    kernel: KernelConfig,
    depth: NonZeroUsize,
    mut param_injection: Option<ParamInjection<'_>>,
) -> (Vec<Vec<Arc<Mutex<Node>>>>, OwnedShape) {
    let mut kernel_layers = vec![];
    let mut kernel_layer_shape = None;
    for depth in 0..depth.get() {
        let param_injection = param_injection
            .as_mut()
            .map(|x| x.name_append(&format!(":depth.{depth}")));
        let (layer, shape) = conv_layer(inputs, stride, kernel.clone(), param_injection);
        if let Some(layer_shape) = &kernel_layer_shape {
            assert_eq!(&shape, layer_shape);
        }
        kernel_layer_shape = Some(shape);
        kernel_layers.push(layer);
    }
    (kernel_layers, kernel_layer_shape.unwrap())
}
