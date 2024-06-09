use std::{
    num::NonZeroUsize,
    sync::{Arc, Mutex},
};

use crate::{node::Node, param::ParamInjection};

use super::{
    bias_node::bias_node,
    weight_node::{weight_node, WeightNodeError},
};

/// ```math
/// f_{w,b} (x) = wx + b
/// ```
///
/// - `lambda`: for regularization
pub fn linear_node(
    input_nodes: Vec<Arc<Mutex<Node>>>,
    initial_weights: Option<Vec<f64>>,
    initial_bias: Option<f64>,
    lambda: Option<f64>,
    param_injection: Option<ParamInjection<'_>>,
) -> Result<Arc<Mutex<Node>>, WeightNodeError> {
    let weight_node = weight_node(input_nodes, initial_weights, lambda)?;
    let weight_node = Arc::new(Mutex::new(weight_node));
    let bias_node = Arc::new(Mutex::new(bias_node(
        Arc::clone(&weight_node),
        initial_bias,
    )));
    if let Some(mut param_injection) = param_injection {
        param_injection
            .name_append(":weights")
            .insert_node(weight_node);
        param_injection
            .name_append(":bias")
            .insert_node(Arc::clone(&bias_node));
    }
    Ok(bias_node)
}

pub fn linear_layer(
    input_nodes: Vec<Arc<Mutex<Node>>>,
    depth: NonZeroUsize,
    initial_weights: Option<&dyn Fn() -> Vec<f64>>,
    initial_bias: Option<&dyn Fn() -> f64>,
    lambda: Option<f64>,
    mut param_injection: Option<ParamInjection<'_>>,
) -> Result<Vec<Arc<Mutex<Node>>>, WeightNodeError> {
    let mut layer = vec![];
    for depth in 0..depth.get() {
        let param_injection = param_injection
            .as_mut()
            .map(|x| x.name_append(&format!(":depth.{depth}")));
        let linear_node = linear_node(
            input_nodes.clone(),
            initial_weights.as_ref().map(|f| f()),
            initial_bias.as_ref().map(|f| f()),
            lambda,
            param_injection,
        )?;
        layer.push(linear_node);
    }
    Ok(layer)
}
