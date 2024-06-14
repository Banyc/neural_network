use conv::{deep_conv_layer, DeepConvLayerConfig};
use max_pooling::max_pooling_layer;

use crate::{
    node::SharedNode,
    param::ParamInjection,
    tensor::{OwnedShape, Tensor},
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
