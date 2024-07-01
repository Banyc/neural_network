use graph::NodeIdx;

use crate::{
    node::{CompNode, GraphBuilder},
    nodes::linear::{linear_layer, LinearLayerConfig},
    param::ParamInjection,
};

use super::activation::Activation;

pub fn dense_layer(
    graph: &mut GraphBuilder,
    inputs: Vec<NodeIdx>,
    config: LinearLayerConfig,
    activation: &Activation,
    param_injection: ParamInjection<'_>,
) -> Vec<CompNode> {
    let depth = config.depth;
    let linear_layer = linear_layer(graph, inputs, config, param_injection).unwrap();
    assert_eq!(linear_layer.len(), depth.get());
    activation.activate(&linear_layer)
}
