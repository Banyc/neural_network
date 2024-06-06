use std::sync::{Arc, Mutex};

use rand::Rng;
use thiserror::Error;

use super::node::{Node, NodeComputation};

/// ```math
/// f_w (x) = wx
/// ```
pub fn weight_node(
    operands: Vec<Arc<Mutex<Node>>>,
    weights: Option<Vec<f64>>,
) -> Result<Node, WeightNodeError> {
    regularized_weight_node(operands, weights, 0.0)
}

pub fn regularized_weight_node(
    operands: Vec<Arc<Mutex<Node>>>,
    mut weights: Option<Vec<f64>>,
    lambda: f64,
) -> Result<Node, WeightNodeError> {
    if let Some(weights) = &weights {
        if operands.len() != weights.len() {
            return Err(WeightNodeError::ParameterSizeNotMatched);
        }
    }
    let computation = WeightNodeComputation { lambda };
    let weights = match weights.take() {
        Some(x) => x,
        None => {
            let op_len = operands.len();
            let weight_bound = 1.0 / (op_len as f64).sqrt();
            let mut rng = rand::thread_rng();
            (0..op_len)
                .map(|_| {
                    let weight: f64 = rng.gen_range(-weight_bound..weight_bound);
                    weight
                })
                .collect()
        }
    };
    let node = Node::new(operands, Arc::new(computation), weights);
    Ok(node)
}

#[derive(Debug)]
struct WeightNodeComputation {
    lambda: f64,
}
impl NodeComputation for WeightNodeComputation {
    fn compute_output(&self, parameters: &[f64], operand_outputs: &[f64], _inputs: &[f64]) -> f64 {
        assert_eq!(operand_outputs.len(), parameters.len());
        weight(operand_outputs, parameters)
    }

    fn compute_gradient_of_this_at_operand(
        &self,
        parameters: &[f64],
        _operand_outputs: &[f64],
    ) -> Vec<f64> {
        weight_derivative(parameters)
    }

    fn compute_gradient_of_this_at_parameter(
        &self,
        _parameters: &[f64],
        operand_outputs: &[f64],
    ) -> Vec<f64> {
        derivative_of_w(operand_outputs)
    }

    fn regularization(&self, parameter: f64) -> f64 {
        derivative_of_l2_regularization(parameter, self.lambda)
    }
}

fn weight(x: &[f64], w: &[f64]) -> f64 {
    assert_eq!(x.len(), w.len());
    (0..w.len()).map(|i| x[i] * w[i]).sum()
}

fn weight_derivative(w: &[f64]) -> Vec<f64> {
    w.to_vec()
}

fn derivative_of_w(x: &[f64]) -> Vec<f64> {
    x.to_vec()
}

fn derivative_of_l2_regularization(w: f64, lambda: f64) -> f64 {
    w * lambda
}

#[derive(Debug, Error)]
pub enum WeightNodeError {
    #[error("Parameter size not matched")]
    ParameterSizeNotMatched,
}

#[cfg(test)]
mod tests {
    use super::{super::input_node::input_node_batch, weight_node};

    #[test]
    fn evaluate() {
        let input_nodes = input_node_batch(3);
        let inputs = vec![1.0, 2.0, 3.0];
        let initial_weights = vec![3.0, 2.0, 1.0];
        let mut weight_node = weight_node(input_nodes, Some(initial_weights)).unwrap();
        let ret = weight_node.evaluate_once(&inputs);
        assert_eq!(ret, 3.0 * 1.0 + 2.0 * 2.0 + 1.0 * 3.0);
    }

    #[test]
    fn gradient_of_this_at_operand() {
        let input_nodes = input_node_batch(3);
        let inputs = vec![1.0, 2.0, 3.0];
        let initial_weights = vec![3.0, 2.0, 1.0];
        let mut weight_node = weight_node(input_nodes, Some(initial_weights)).unwrap();
        weight_node.evaluate_once(&inputs);
        let ret = weight_node.gradient_of_this_at_operand().unwrap();
        assert_eq!(&ret, &[3.0, 2.0, 1.0]);
    }

    #[test]
    fn gradient_of_this_at_parameter() {
        let input_nodes = input_node_batch(3);
        let inputs = vec![1.0, 2.0, 3.0];
        let initial_weights = vec![3.0, 2.0, 1.0];
        let mut weight_node = weight_node(input_nodes, Some(initial_weights)).unwrap();
        weight_node.evaluate_once(&inputs);
        let ret = weight_node.gradient_of_this_at_parameter().unwrap();
        assert_eq!(&ret, &[1.0, 2.0, 3.0]);
    }
}
