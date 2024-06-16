use std::num::NonZeroUsize;

use crate::{
    layers::kernel::{kernel_layer, KernelLayerConfig, KernelParams},
    node::SharedNode,
    nodes::linear::linear_node,
    param::ParamInjection,
    tensor::{append_tensors, non_zero_to_shape, OwnedShape, Shape, Tensor},
};

#[derive(Clone)]
pub struct ConvLayerConfig<'a> {
    pub kernel_layer: KernelLayerConfig<'a>,
    pub lambda: Option<f64>,
}
pub fn conv_layer(
    inputs: Tensor<'_, SharedNode>,
    config: ConvLayerConfig<'_>,
    mut param_injection: ParamInjection<'_>,
) -> (Vec<SharedNode>, OwnedShape) {
    let create_filter = |params: KernelParams| -> SharedNode {
        let param_injection = param_injection.name_append(":kernel");
        let feature_node = linear_node(params.inputs, config.lambda, param_injection);
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
    mut param_injection: ParamInjection<'_>,
) -> (Vec<SharedNode>, OwnedShape) {
    let mut kernel_layers = vec![];
    let mut kernel_layer_shape = None;
    for depth in 0..config.depth.get() {
        let param_injection = param_injection.name_append(&format!(":depth.{depth}"));
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
