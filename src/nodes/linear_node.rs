use std::sync::{Arc, Mutex};

use crate::node::Node;

use super::{
    bias_node::bias_node,
    weight_node::{regularized_weight_node, weight_node, WeightNodeError},
};

/// ```math
/// f_{w,b} (x) = wx + b
/// ```
pub fn linear_node(
    input_nodes: Vec<Arc<Mutex<Node>>>,
    initial_weights: Option<Vec<f64>>,
    initial_bias: Option<f64>,
) -> Result<Node, WeightNodeError> {
    let weight_node = weight_node(input_nodes, initial_weights)?;
    let bias_node = bias_node(Arc::new(Mutex::new(weight_node)), initial_bias);
    Ok(bias_node)
}

pub fn regularized_linear_node(
    input_nodes: Vec<Arc<Mutex<Node>>>,
    initial_weights: Option<Vec<f64>>,
    initial_bias: Option<f64>,
    lambda: f64,
) -> Result<Node, WeightNodeError> {
    let weight_node = regularized_weight_node(input_nodes, initial_weights, lambda)?;
    let bias_node = bias_node(Arc::new(Mutex::new(weight_node)), initial_bias);
    Ok(bias_node)
}
