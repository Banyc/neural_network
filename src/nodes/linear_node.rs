use std::sync::{Arc, Mutex};

use crate::{node::Node, param::ParamInjector};

use super::{
    bias_node::bias_node,
    weight_node::{weight_node, WeightNodeError},
};

#[derive(Debug)]
pub struct ParamInjection<'a> {
    pub injector: &'a mut ParamInjector,
    pub weight_name: String,
    pub bias_name: String,
}

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
    param_injection: Option<&mut ParamInjection<'_>>,
) -> Result<Arc<Mutex<Node>>, WeightNodeError> {
    let weight_node = weight_node(input_nodes, initial_weights, lambda)?;
    let weight_node = Arc::new(Mutex::new(weight_node));
    let bias_node = Arc::new(Mutex::new(bias_node(
        Arc::clone(&weight_node),
        initial_bias,
    )));
    if let Some(param_injection) = param_injection {
        param_injection
            .injector
            .insert_node(param_injection.weight_name.clone(), weight_node);
        param_injection
            .injector
            .insert_node(param_injection.bias_name.clone(), Arc::clone(&bias_node));
    }
    Ok(bias_node)
}
