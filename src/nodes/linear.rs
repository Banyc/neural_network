use std::num::NonZeroUsize;

use graph::NodeIdx;
use vec_seg::SegKey;

use crate::{node::GraphBuilder, param::ParamInjection};

use super::{
    bias::{bias_node, default_bias},
    weights::{rnd_weights, weight_node, WeightNodeError},
};

/// ```math
/// f_{w,b} (x) = wx + b
/// ```
///
/// - `lambda`: for regularization
pub fn linear_node(
    graph: &mut GraphBuilder,
    input_nodes: Vec<NodeIdx>,
    lambda: Option<f64>,
    mut param_injection: ParamInjection<'_>,
) -> Result<NodeIdx, WeightNodeError> {
    let weights = param_injection
        .name_append(":weights")
        .get_or_create_params(|| rnd_weights(input_nodes.len()).into_iter());
    let bias = param_injection
        .name_append(":bias")
        .get_or_create_params(|| [default_bias()].into_iter());
    linear_node_manual(graph, input_nodes, lambda, weights, bias)
}
pub fn linear_node_manual(
    graph: &mut GraphBuilder,
    input_nodes: Vec<NodeIdx>,
    lambda: Option<f64>,
    weights: SegKey,
    bias: SegKey,
) -> Result<NodeIdx, WeightNodeError> {
    let weight_node = graph.insert_node(weight_node(input_nodes, weights, lambda)?);
    let bias_node = graph.insert_node(bias_node(weight_node, bias));
    Ok(bias_node)
}

pub struct LinearLayerConfig {
    pub depth: NonZeroUsize,
    pub lambda: Option<f64>,
}
pub fn linear_layer(
    graph: &mut GraphBuilder,
    input_nodes: Vec<NodeIdx>,
    config: LinearLayerConfig,
    mut param_injection: ParamInjection<'_>,
) -> Result<Vec<NodeIdx>, WeightNodeError> {
    let mut layer = vec![];
    for depth in 0..config.depth.get() {
        let param_injection = param_injection.name_append(&format!(":depth.{depth}"));
        let linear_node = linear_node(graph, input_nodes.clone(), config.lambda, param_injection)?;
        layer.push(linear_node);
    }
    Ok(layer)
}
