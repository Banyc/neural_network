use std::num::NonZeroUsize;

use conv::{deep_conv_layer, DeepConvLayerConfig};
use pooling::max_pooling_layer;

use crate::{
    layers::residual::residual_layer,
    node::SharedNode,
    param::ParamInjection,
    tensor::{OwnedShape, Shape, Tensor},
};

use super::{activation::Activation, kernel::KernelLayerConfig, norm::Normalization};

pub mod conv;
pub mod pooling;

pub fn conv_max_pooling_layer(
    inputs: Tensor<'_, SharedNode>,
    conv: DeepConvLayerConfig<'_>,
    max_pooling: KernelLayerConfig<'_>,
    activation: &Activation,
    normalization: Option<Normalization>,
    mut param_injection: ParamInjection<'_>,
) -> (Vec<SharedNode>, OwnedShape) {
    let (conv_layer, shape) = {
        let param_injection = param_injection.name_append(":conv");
        deep_conv_layer(inputs, conv, param_injection)
    };
    let activation_layer = activation.activate(&conv_layer);
    let normalization_layer = if let Some(norm) = normalization {
        let param_injection = param_injection.name_append(":norm");
        norm.normalize(activation_layer, param_injection)
    } else {
        activation_layer
    };
    {
        let inputs = Tensor::new(&normalization_layer, &shape).unwrap();
        max_pooling_layer(inputs, max_pooling)
    }
}

#[derive(Clone)]
pub struct ResidualConvConfig<'a> {
    pub first_conv: DeepConvLayerConfig<'a>,
    pub following_conv: DeepConvLayerConfig<'a>,
    pub num_conv_layers: NonZeroUsize,
}
pub fn residual_conv_layers(
    inputs: Vec<SharedNode>,
    inputs_shape: &Shape,
    activation: &Activation,
    normalization: Normalization,
    config: ResidualConvConfig<'_>,
    mut param_injection: ParamInjection<'_>,
) -> (Vec<SharedNode>, OwnedShape) {
    let mut prev_layer: Option<(Vec<SharedNode>, OwnedShape)> = None;

    for i in 0..config.num_conv_layers.get() {
        let imm_inputs = match &prev_layer {
            Some((layer, shape)) => {
                assert!(i != 0);
                Tensor::new(layer, shape).unwrap()
            }
            None => {
                assert!(i == 0);
                Tensor::new(&inputs, inputs_shape).unwrap()
            }
        };

        let is_end = i + 1 == config.num_conv_layers.get();

        let (conv_layer, shape) = {
            let param_injection = param_injection.name_append(&format!(":conv.{i}"));
            let config = if i == 0 {
                &config.first_conv
            } else {
                &config.following_conv
            };
            deep_conv_layer(imm_inputs, config.clone(), param_injection)
        };
        let res_layer = if is_end {
            let param_injection = param_injection.name_append(&format!(":res.{i}"));
            residual_layer(conv_layer, inputs.clone(), param_injection)
        } else {
            conv_layer
        };
        let act_layer = activation.activate(&res_layer);
        let layer = {
            let param_injection = param_injection.name_append(&format!(":norm.{i}"));
            normalization.clone().normalize(act_layer, param_injection)
        };
        prev_layer = Some((layer, shape));
    }

    prev_layer.unwrap()
}
