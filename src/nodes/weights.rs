use rand::Rng;
use thiserror::Error;

use crate::{
    computation::{NodeBackpropagationComputation, NodeComputation, NodeScalarComputation},
    mut_cell::MutCell,
    node::{Node, SharedNode},
    param::SharedParams,
    ref_ctr::RefCtr,
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
    operands: Vec<SharedNode>,
    weights: SharedParams,
    lambda: Option<f64>,
) -> Result<Node, WeightNodeError> {
    if operands.len() != weights.borrow().len() {
        return Err(WeightNodeError::ParameterSizeNotMatched);
    }
    let computation = WeightNodeComputation {
        lambda: lambda.unwrap_or(0.),
    };
    let node = Node::new(
        operands,
        RefCtr::new(MutCell::new(NodeComputation::Scalar(Box::new(computation)))),
        weights,
    );
    Ok(node)
}

#[derive(Debug)]
struct WeightNodeComputation {
    lambda: f64,
}
impl NodeScalarComputation for WeightNodeComputation {
    fn compute_output(
        &self,
        parameters: &[f64],
        operand_outputs: &[f64],
        _graph_inputs: &[f64],
    ) -> f64 {
        assert_eq!(operand_outputs.len(), parameters.len());
        weight(operand_outputs, parameters)
    }
}
impl NodeBackpropagationComputation for WeightNodeComputation {
    fn compute_gradient_of_this_at_operand(
        &self,
        parameters: &[f64],
        operand_outputs: &[f64],
        buf: Vec<f64>,
    ) -> Vec<f64> {
        assert_eq!(operand_outputs.len(), parameters.len());
        weight_derivative(parameters, buf)
    }

    fn compute_gradient_of_this_at_parameter(
        &self,
        parameters: &[f64],
        operand_outputs: &[f64],
        buf: Vec<f64>,
    ) -> Vec<f64> {
        assert_eq!(operand_outputs.len(), parameters.len());
        derivative_of_w(operand_outputs, buf)
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

fn weight_derivative(w: &[f64], mut buf: Vec<f64>) -> Vec<f64> {
    buf.extend(w);
    buf
}

fn derivative_of_w(x: &[f64], mut buf: Vec<f64>) -> Vec<f64> {
    buf.extend(x);
    buf
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
    use super::*;

    use crate::{
        computation::ComputationMode,
        node::NodeContext,
        nodes::input::{input_node_batch, InputNodeBatchParams},
    };

    #[test]
    fn evaluate() {
        let input_nodes = input_node_batch(InputNodeBatchParams { start: 0, len: 3 });
        let inputs = vec![1.0, 2.0, 3.0];
        let initial_weights = vec![3.0, 2.0, 1.0];
        let initial_weights = RefCtr::new(MutCell::new(initial_weights));
        let mut weight_node = weight_node(input_nodes, initial_weights, None).unwrap();
        let mut cx = NodeContext::new();
        weight_node.evaluate_once(&[&inputs], &mut cx, ComputationMode::Inference);
        let output = weight_node.output().unwrap()[0];
        assert_eq!(output, 3.0 * 1.0 + 2.0 * 2.0 + 1.0 * 3.0);
    }

    #[test]
    fn gradient_of_this_at_operand() {
        let input_nodes = input_node_batch(InputNodeBatchParams { start: 0, len: 3 });
        let inputs = vec![1.0, 2.0, 3.0];
        let initial_weights = vec![3.0, 2.0, 1.0];
        let initial_weights = RefCtr::new(MutCell::new(initial_weights));
        let mut weight_node = weight_node(input_nodes, initial_weights, None).unwrap();
        let mut cx = NodeContext::new();
        weight_node.evaluate_once(&[&inputs], &mut cx, ComputationMode::Inference);
        let batch_index = 0;
        let ret = weight_node
            .gradient_of_this_at_operand(batch_index, &weight_node.parameters().borrow(), &mut cx)
            .unwrap();
        assert_eq!(&ret, &[3.0, 2.0, 1.0]);
    }

    #[test]
    fn gradient_of_this_at_parameter() {
        let input_nodes = input_node_batch(InputNodeBatchParams { start: 0, len: 3 });
        let inputs = vec![1.0, 2.0, 3.0];
        let initial_weights = vec![3.0, 2.0, 1.0];
        let initial_weights = RefCtr::new(MutCell::new(initial_weights));
        let mut weight_node = weight_node(input_nodes, initial_weights, None).unwrap();
        let mut cx = NodeContext::new();
        weight_node.evaluate_once(&[&inputs], &mut cx, ComputationMode::Inference);
        let batch_index = 0;
        let ret = weight_node
            .gradient_of_this_at_parameter(batch_index, &weight_node.parameters().borrow(), &mut cx)
            .unwrap();
        assert_eq!(&ret, &[1.0, 2.0, 3.0]);
    }
}
