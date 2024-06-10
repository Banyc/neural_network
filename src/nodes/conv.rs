use std::{
    num::NonZeroUsize,
    sync::{Arc, Mutex},
};

use crate::{
    node::SharedNode,
    param::{ParamInjection, SharedParams},
    tensor::{NonZeroShape, OwnedShape, Stride, Tensor},
};

use super::{
    bias::default_bias,
    kernel::{kernel_layer, KernelParams},
    linear::linear_node,
    weights::rnd_weights,
};

#[derive(Clone)]
pub struct KernelConfig<'a> {
    pub shape: &'a NonZeroShape,
    pub initial_weights: Option<&'a dyn Fn() -> SharedParams>,
    pub initial_bias: Option<&'a dyn Fn() -> SharedParams>,
    pub lambda: Option<f64>,
}

pub fn conv_layer(
    inputs: Tensor<'_, SharedNode>,
    stride: &Stride,
    kernel: KernelConfig,
    param_injection: Option<ParamInjection<'_>>,
) -> (Vec<SharedNode>, OwnedShape) {
    let weights = kernel
        .initial_weights
        .as_ref()
        .map(|f| f())
        .unwrap_or_else(|| {
            let num_kernel_params = kernel.shape.iter().map(|x| x.get()).product();
            let weights = rnd_weights(num_kernel_params);
            Arc::new(Mutex::new(weights))
        });
    let bias = kernel
        .initial_bias
        .as_ref()
        .map(|f| f())
        .unwrap_or_else(|| Arc::new(Mutex::new(vec![default_bias()])));
    if let Some(mut param_injection) = param_injection {
        param_injection
            .name_append(":weights")
            .insert_params(Arc::clone(&weights));
        param_injection
            .name_append(":bias")
            .insert_params(Arc::clone(&bias));
    }

    let create_filter = |params: KernelParams| -> SharedNode {
        let feature_node = linear_node(
            params.inputs,
            Some(Arc::clone(&weights)),
            Some(Arc::clone(&bias)),
            kernel.lambda,
            None,
        );
        feature_node.unwrap()
    };
    kernel_layer(inputs, stride, kernel.shape, create_filter)
}

pub fn deep_conv_layer(
    inputs: Tensor<'_, SharedNode>,
    stride: &Stride,
    kernel: KernelConfig,
    depth: NonZeroUsize,
    mut param_injection: Option<ParamInjection<'_>>,
) -> (Vec<Vec<SharedNode>>, OwnedShape) {
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
