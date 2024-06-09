use std::sync::{Arc, Mutex};

use rand::Rng;
use thiserror::Error;

use crate::{
    node::{Node, NodeComputation},
    param::SharedParams,
};

/// ```math
/// X \sim U(-\frac{1}{n^2}, \frac{1}{n^2})
/// ```
pub fn rnd_weights(op_len: usize) -> Vec<f64> {
    if op_len == 0 {
        return vec![];
    }
    let weight_bound = 1.0 / (op_len as f64).sqrt();
    let mut rng = rand::thread_rng();
    (0..op_len)
        .map(|_| {
            let weight: f64 = rng.gen_range(-weight_bound..weight_bound);
            weight
        })
        .collect()
}

/// ```math
/// f_w (x) = wx
/// ```
///
/// - `lambda`: for regularization
pub fn weight_node(
    operands: Vec<Arc<Mutex<Node>>>,
    mut weights: Option<SharedParams>,
    lambda: Option<f64>,
) -> Result<Node, WeightNodeError> {
    if let Some(weights) = &weights {
        if operands.len() != weights.lock().unwrap().len() {
            return Err(WeightNodeError::ParameterSizeNotMatched);
        }
    }
    let computation = WeightNodeComputation {
        lambda: lambda.unwrap_or(0.),
    };
    let weights = match weights.take() {
        Some(x) => x,
        None => Arc::new(Mutex::new(rnd_weights(operands.len()))),
    };
    let node = Node::new(operands, Arc::new(computation), weights);
    Ok(node)
}

#[derive(Debug)]
struct WeightNodeComputation {
    lambda: f64,
}
impl NodeComputation for WeightNodeComputation {
    fn compute_output(
        &self,
        parameters: &[f64],
        operand_outputs: &[f64],
        _graph_inputs: &[f64],
    ) -> f64 {
        assert_eq!(operand_outputs.len(), parameters.len());
        weight(operand_outputs, parameters)
    }

    fn compute_gradient_of_this_at_operand(
        &self,
        parameters: &[f64],
        operand_outputs: &[f64],
    ) -> Vec<f64> {
        assert_eq!(operand_outputs.len(), parameters.len());
        weight_derivative(parameters)
    }

    fn compute_gradient_of_this_at_parameter(
        &self,
        parameters: &[f64],
        operand_outputs: &[f64],
    ) -> Vec<f64> {
        assert_eq!(operand_outputs.len(), parameters.len());
        derivative_of_w(operand_outputs)
    }

    fn regularization(&self, parameter: f64) -> f64 {
        derivative_of_l2_regularization(parameter, self.lambda)
    }
}

fn weight(x: &[f64], w: &[f64]) -> f64 {
    assert_eq!(x.len(), w.len());
    x.iter()
        .copied()
        .zip(w.iter().copied())
        .map(|(x, w)| x * w)
        .sum()
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
    use std::sync::{Arc, Mutex};

    use crate::nodes::input_node::InputNodeBatchParams;

    use super::{super::input_node::input_node_batch, weight_node};

    #[test]
    fn evaluate() {
        let input_nodes = input_node_batch(InputNodeBatchParams { start: 0, len: 3 });
        let inputs = vec![1.0, 2.0, 3.0];
        let initial_weights = vec![3.0, 2.0, 1.0];
        let initial_weights = Arc::new(Mutex::new(initial_weights));
        let mut weight_node = weight_node(input_nodes, Some(initial_weights), None).unwrap();
        let batch_index = 0;
        let ret = weight_node.evaluate_once(&inputs, batch_index);
        assert_eq!(ret, 3.0 * 1.0 + 2.0 * 2.0 + 1.0 * 3.0);
    }

    #[test]
    fn gradient_of_this_at_operand() {
        let input_nodes = input_node_batch(InputNodeBatchParams { start: 0, len: 3 });
        let inputs = vec![1.0, 2.0, 3.0];
        let initial_weights = vec![3.0, 2.0, 1.0];
        let initial_weights = Arc::new(Mutex::new(initial_weights));
        let mut weight_node = weight_node(input_nodes, Some(initial_weights), None).unwrap();
        let batch_index = 0;
        weight_node.evaluate_once(&inputs, batch_index);
        let ret = weight_node
            .gradient_of_this_at_operand(batch_index, &weight_node.parameters().lock().unwrap())
            .unwrap();
        assert_eq!(&ret, &[3.0, 2.0, 1.0]);
    }

    #[test]
    fn gradient_of_this_at_parameter() {
        let input_nodes = input_node_batch(InputNodeBatchParams { start: 0, len: 3 });
        let inputs = vec![1.0, 2.0, 3.0];
        let initial_weights = vec![3.0, 2.0, 1.0];
        let initial_weights = Arc::new(Mutex::new(initial_weights));
        let mut weight_node = weight_node(input_nodes, Some(initial_weights), None).unwrap();
        let batch_index = 0;
        weight_node.evaluate_once(&inputs, batch_index);
        let ret = weight_node
            .gradient_of_this_at_parameter(batch_index, &weight_node.parameters().lock().unwrap())
            .unwrap();
        assert_eq!(&ret, &[1.0, 2.0, 3.0]);
    }
}
