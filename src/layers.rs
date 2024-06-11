use std::{cell::RefCell, sync::Arc};

use crate::{
    node::SharedNode,
    nodes::{
        conv::{deep_conv_layer, DeepConvLayerConfig},
        kernel::KernelLayerConfig,
        linear::{linear_layer, LinearLayerConfig},
        max_pooling::max_pooling_layer,
        relu::relu_node,
        swish::swish_node,
    },
    param::ParamInjection,
    tensor::{OwnedShape, Tensor},
};

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

pub fn dense_layer(
    inputs: Vec<SharedNode>,
    config: LinearLayerConfig,
    activation: &Activation,
    param_injection: Option<ParamInjection<'_>>,
) -> Vec<SharedNode> {
    let depth = config.depth;
    let linear_layer = linear_layer(inputs, config, param_injection).unwrap();
    assert_eq!(linear_layer.len(), depth.get());
    activation.activate(&linear_layer)
}

#[derive(Debug, Clone)]
pub enum Activation {
    ReLu,
    Swish,
}
impl Activation {
    pub fn activate(&self, inputs: &[SharedNode]) -> Vec<SharedNode> {
        inputs
            .iter()
            .map(Arc::clone)
            .map(|x| match self {
                Activation::ReLu => relu_node(x),
                Activation::Swish => swish_node(x),
            })
            .map(|x| Arc::new(RefCell::new(x)))
            .collect::<Vec<SharedNode>>()
    }
}
