use std::num::NonZeroUsize;

use conv::{deep_conv_layer, DeepConvLayerConfig};
use max_pooling::max_pooling_layer;

use crate::{
    layers::residual::residual_layer,
    node::SharedNode,
    param::ParamInjection,
    tensor::{OwnedShape, Shape, Tensor},
};

use super::{activation::Activation, kernel::KernelLayerConfig};

pub mod conv;
pub mod max_pooling;

pub fn conv_max_pooling_layer(
    inputs: Tensor<'_, SharedNode>,
    conv: DeepConvLayerConfig<'_>,
    max_pooling: KernelLayerConfig<'_>,
    activation: &Activation,
    param_injection: Option<ParamInjection<'_>>,
) -> (Vec<SharedNode>, OwnedShape) {
    let (conv_layer, shape) = deep_conv_layer(inputs, conv, param_injection);
    let activation_layer = activation.activate(&conv_layer);
    {
        let inputs = Tensor::new(&activation_layer, &shape).unwrap();
        max_pooling_layer(inputs, max_pooling)
    }
}

#[derive(Clone)]
pub struct ResidualConvConfig<'a> {
    pub conv: DeepConvLayerConfig<'a>,
    pub num_conv_layers: NonZeroUsize,
}
pub fn residual_conv_layers(
    inputs: Vec<SharedNode>,
    inputs_shape: &Shape,
    activation: &Activation,
    config: ResidualConvConfig<'_>,
    mut param_injection: Option<ParamInjection<'_>>,
) -> (Vec<SharedNode>, OwnedShape) {
    let mut prev_layer: Option<(Vec<SharedNode>, OwnedShape)> = None;

    for i in 0..config.num_conv_layers.get() {
        let imm_inputs = match &prev_layer {
            Some((layer, shape)) => Tensor::new(layer, shape).unwrap(),
            None => Tensor::new(&inputs, inputs_shape).unwrap(),
        };

        let is_end = i + 1 == config.num_conv_layers.get();

        let conv_param_injection = param_injection
            .as_mut()
            .map(|x| x.name_append(&format!(":conv.{i}")));
        let (conv_layer, shape) =
            deep_conv_layer(imm_inputs, config.conv.clone(), conv_param_injection);
        let res_layer = if is_end {
            let param_injection = param_injection
                .as_mut()
                .map(|x| x.name_append(&format!(":res.{i}")));
            residual_layer(conv_layer, inputs.clone(), param_injection)
        } else {
            conv_layer
        };
        let layer = activation.activate(&res_layer);
        prev_layer = Some((layer, shape));
    }

    prev_layer.unwrap()
}
