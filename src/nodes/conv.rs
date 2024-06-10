use std::{cell::RefCell, num::NonZeroUsize, sync::Arc};

use crate::{
    node::SharedNode,
    param::{ParamInjection, SharedParams},
    tensor::{append_tensors, non_zero_to_shape, OwnedShape, Shape, Tensor},
};

use super::{
    bias::default_bias,
    kernel::{kernel_layer, KernelLayerConfig, KernelParams},
    linear::linear_node,
    weights::rnd_weights,
};

#[derive(Clone)]
pub struct ConvLayerConfig<'a> {
    pub kernel_layer: KernelLayerConfig<'a>,
    pub initial_weights: Option<&'a dyn Fn() -> SharedParams>,
    pub initial_bias: Option<&'a dyn Fn() -> SharedParams>,
    pub lambda: Option<f64>,
}
pub fn conv_layer(
    inputs: Tensor<'_, SharedNode>,
    config: ConvLayerConfig<'_>,
    param_injection: Option<ParamInjection<'_>>,
) -> (Vec<SharedNode>, OwnedShape) {
    let weights = config
        .initial_weights
        .as_ref()
        .map(|f| f())
        .unwrap_or_else(|| {
            let num_kernel_params = config
                .kernel_layer
                .kernel_shape
                .iter()
                .map(|x| x.get())
                .product();
            let weights = rnd_weights(num_kernel_params);
            Arc::new(RefCell::new(weights))
        });
    let bias = config
        .initial_bias
        .as_ref()
        .map(|f| f())
        .unwrap_or_else(|| Arc::new(RefCell::new(vec![default_bias()])));
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
            config.lambda,
            None,
        );
        feature_node.unwrap()
    };
    kernel_layer(inputs, config.kernel_layer, create_filter)
}

#[derive(Clone)]
pub struct DeepConvLayerConfig<'a> {
    pub depth: NonZeroUsize,
    pub conv: ConvLayerConfig<'a>,
    pub assert_output_shape: Option<&'a Shape>,
}
pub fn deep_conv_layer(
    inputs: Tensor<'_, SharedNode>,
    config: DeepConvLayerConfig<'_>,
    mut param_injection: Option<ParamInjection<'_>>,
) -> (Vec<SharedNode>, OwnedShape) {
    let mut kernel_layers = vec![];
    let mut kernel_layer_shape = None;
    for depth in 0..config.depth.get() {
        let param_injection = param_injection
            .as_mut()
            .map(|x| x.name_append(&format!(":depth.{depth}")));
        let (layer, shape) = conv_layer(inputs, config.conv.clone(), param_injection);
        if let Some(layer_shape) = &kernel_layer_shape {
            assert_eq!(&shape, layer_shape);
        }
        kernel_layer_shape = Some(shape);
        kernel_layers.push(layer);
    }
    let (layer, shape) = append_tensors(kernel_layers, &kernel_layer_shape.unwrap());
    let shape = non_zero_to_shape(&shape);
    if let Some(assert_output_shape) = config.assert_output_shape {
        assert_eq!(assert_output_shape, shape);
    }
    (layer, shape)
}
