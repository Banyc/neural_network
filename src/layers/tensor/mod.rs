use std::num::NonZeroUsize;

use conv::{deep_conv_layer, DeepConvLayerConfig};
use graph::NodeIdx;
use pooling::max_pooling_layer;

use crate::{
    layers::residual::residual_layer,
    node::GraphBuilder,
    param::ParamInjection,
    tensor::{OwnedShape, Shape, Tensor},
};

use super::{activation::Activation, kernel::KernelLayerConfig, norm::Normalization};

pub mod conv;
pub mod pooling;

pub fn conv_max_pooling_layer(
    graph: &mut GraphBuilder,
    inputs: Tensor<'_, NodeIdx>,
    conv: DeepConvLayerConfig<'_>,
    max_pooling: KernelLayerConfig<'_>,
    activation: &Activation,
    normalization: Option<Normalization>,
    mut param_injection: ParamInjection<'_>,
) -> (Vec<NodeIdx>, OwnedShape) {
    let (conv_layer, shape) = {
        let param_injection = param_injection.name_append(":conv");
        deep_conv_layer(graph, inputs, conv, param_injection)
    };
    let activation_layer = graph.insert_nodes(activation.activate(&conv_layer));
    let normalization_layer = if let Some(norm) = normalization {
        let param_injection = param_injection.name_append(":norm");
        let norm = norm.normalize(graph, activation_layer, param_injection);
        graph.insert_nodes(norm)
    } else {
        activation_layer
    };
    {
        let inputs = Tensor::new(&normalization_layer, &shape).unwrap();
        max_pooling_layer(graph, inputs, max_pooling)
    }
}

#[derive(Clone)]
pub struct ResidualConvConfig<'a> {
    pub first_conv: DeepConvLayerConfig<'a>,
    pub following_conv: DeepConvLayerConfig<'a>,
    pub num_conv_layers: NonZeroUsize,
}
pub fn residual_conv_layers(
    graph: &mut GraphBuilder,
    inputs: Vec<NodeIdx>,
    inputs_shape: &Shape,
    activation: &Activation,
    normalization: Normalization,
    config: ResidualConvConfig<'_>,
    mut param_injection: ParamInjection<'_>,
) -> (Vec<NodeIdx>, OwnedShape) {
    let mut prev_layer: Option<(Vec<NodeIdx>, OwnedShape)> = None;

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
            deep_conv_layer(graph, imm_inputs, config.clone(), param_injection)
        };
        let res_layer = if is_end {
            let param_injection = param_injection.name_append(&format!(":res.{i}"));
            residual_layer(graph, conv_layer, inputs.clone(), param_injection)
        } else {
            conv_layer
        };
        let act_layer = graph.insert_nodes(activation.activate(&res_layer));
        let layer = {
            let param_injection = param_injection.name_append(&format!(":norm.{i}"));
            let norm = normalization
                .clone()
                .normalize(graph, act_layer, param_injection);
            graph.insert_nodes(norm)
        };
        prev_layer = Some((layer, shape));
    }

    prev_layer.unwrap()
}
