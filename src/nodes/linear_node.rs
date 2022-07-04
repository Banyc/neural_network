use std::sync::{Arc, Mutex};

use super::{
    bias_node::bias_node,
    node::GeneralNode,
    weight_node::{weight_node, WeightNodeError},
};

pub fn linear_node(
    input_nodes: Vec<Arc<Mutex<GeneralNode>>>,
    initial_weights: Option<Vec<f64>>,
    initial_bias: Option<f64>,
) -> Result<GeneralNode, WeightNodeError> {
    let weight_node = weight_node(input_nodes, initial_weights)?;
    let bias_node = bias_node(Arc::new(Mutex::new(weight_node)), initial_bias);
    Ok(bias_node)
}
