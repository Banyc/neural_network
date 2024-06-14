use crate::{
    node::SharedNode,
    nodes::linear::{linear_layer, LinearLayerConfig},
    param::ParamInjection,
};

use super::activation::Activation;

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
