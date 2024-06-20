use std::num::NonZeroUsize;

use crate::{
    mut_cell::MutCell,
    node::SharedNode,
    param::{ParamInjection, SharedParams},
    ref_ctr::RefCtr,
};

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
    input_nodes: Vec<SharedNode>,
    lambda: Option<f64>,
    mut param_injection: ParamInjection<'_>,
) -> Result<SharedNode, WeightNodeError> {
    let weights = param_injection
        .name_append(":weights")
        .get_or_create_params(|| RefCtr::new(MutCell::new(rnd_weights(input_nodes.len()))));
    let bias = param_injection
        .name_append(":bias")
        .get_or_create_params(|| RefCtr::new(MutCell::new(vec![default_bias()])));
    linear_node_manual(input_nodes, lambda, weights, bias)
}
pub fn linear_node_manual(
    input_nodes: Vec<SharedNode>,
    lambda: Option<f64>,
    weights: SharedParams,
    bias: SharedParams,
) -> Result<SharedNode, WeightNodeError> {
    let weight_node = weight_node(input_nodes, weights, lambda)?;
    let weight_node = RefCtr::new(MutCell::new(weight_node));
    let bias_node = bias_node(RefCtr::clone(&weight_node), bias);
    let bias_node = RefCtr::new(MutCell::new(bias_node));
    Ok(bias_node)
}

pub struct LinearLayerConfig {
    pub depth: NonZeroUsize,
    pub lambda: Option<f64>,
}
pub fn linear_layer(
    input_nodes: Vec<SharedNode>,
    config: LinearLayerConfig,
    mut param_injection: ParamInjection<'_>,
) -> Result<Vec<SharedNode>, WeightNodeError> {
    let mut layer = vec![];
    for depth in 0..config.depth.get() {
        let param_injection = param_injection.name_append(&format!(":depth.{depth}"));
        let linear_node = linear_node(input_nodes.clone(), config.lambda, param_injection)?;
        layer.push(linear_node);
    }
    Ok(layer)
}
