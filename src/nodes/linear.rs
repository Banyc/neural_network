use std::{
    num::NonZeroUsize,
    sync::{Arc, Mutex},
};

use crate::{
    node::SharedNode,
    param::{ParamInjection, SharedParams},
};

use super::{
    bias::bias_node,
    weights::{weight_node, WeightNodeError},
};

/// ```math
/// f_{w,b} (x) = wx + b
/// ```
///
/// - `lambda`: for regularization
pub fn linear_node(
    input_nodes: Vec<SharedNode>,
    initial_weights: Option<SharedParams>,
    initial_bias: Option<SharedParams>,
    lambda: Option<f64>,
    param_injection: Option<ParamInjection<'_>>,
) -> Result<SharedNode, WeightNodeError> {
    let weight_node = weight_node(input_nodes, initial_weights, lambda)?;
    let weights = Arc::clone(weight_node.parameters());
    let weight_node = Arc::new(Mutex::new(weight_node));
    let bias_node = bias_node(Arc::clone(&weight_node), initial_bias);
    let bias = Arc::clone(bias_node.parameters());
    let bias_node = Arc::new(Mutex::new(bias_node));
    if let Some(mut param_injection) = param_injection {
        param_injection
            .name_append(":weights")
            .insert_params(weights);
        param_injection.name_append(":bias").insert_params(bias);
    }
    Ok(bias_node)
}

pub struct LinearLayerConfig<'a> {
    pub depth: NonZeroUsize,
    pub initial_weights: Option<&'a dyn Fn() -> SharedParams>,
    pub initial_bias: Option<&'a dyn Fn() -> SharedParams>,
    pub lambda: Option<f64>,
}
pub fn linear_layer(
    input_nodes: Vec<SharedNode>,
    config: LinearLayerConfig,
    mut param_injection: Option<ParamInjection<'_>>,
) -> Result<Vec<SharedNode>, WeightNodeError> {
    let mut layer = vec![];
    for depth in 0..config.depth.get() {
        let param_injection = param_injection
            .as_mut()
            .map(|x| x.name_append(&format!(":depth.{depth}")));
        let linear_node = linear_node(
            input_nodes.clone(),
            config.initial_weights.as_ref().map(|f| f()),
            config.initial_bias.as_ref().map(|f| f()),
            config.lambda,
            param_injection,
        )?;
        layer.push(linear_node);
    }
    Ok(layer)
}
