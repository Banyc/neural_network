use std::sync::Arc;

use crate::{
    computation::{NodeBackpropagationComputation, NodeComputation, NodeScalarComputation},
    mut_cell::MutCell,
    node::{Node, SharedNode},
    param::empty_shared_params,
};

/// ```math
/// f(x) = 1 + e^x
/// ```
pub fn sigmoid_node(operand: SharedNode) -> Node {
    let computation = SigmoidNodeComputation {};
    Node::new(
        vec![operand],
        Arc::new(MutCell::new(NodeComputation::Scalar(Box::new(computation)))),
        empty_shared_params(),
    )
}

#[derive(Debug)]
struct SigmoidNodeComputation {}
impl NodeScalarComputation for SigmoidNodeComputation {
    fn compute_output(
        &self,
        parameters: &[f64],
        operand_outputs: &[f64],
        _graph_inputs: &[f64],
    ) -> f64 {
        assert!(parameters.is_empty());
        assert_eq!(operand_outputs.len(), 1);
        sigmoid(operand_outputs[0])
    }
}
impl NodeBackpropagationComputation for SigmoidNodeComputation {
    fn compute_gradient_of_this_at_operand(
        &self,
        parameters: &[f64],
        operand_outputs: &[f64],
        mut buf: Vec<f64>,
    ) -> Vec<f64> {
        assert!(parameters.is_empty());
        assert_eq!(operand_outputs.len(), 1);
        buf.extend([sigmoid_derivative(operand_outputs[0])]);
        buf
    }

    fn compute_gradient_of_this_at_parameter(
        &self,
        parameters: &[f64],
        operand_outputs: &[f64],
        buf: Vec<f64>,
    ) -> Vec<f64> {
        assert!(parameters.is_empty());
        assert_eq!(operand_outputs.len(), 1);
        buf
    }
}

pub fn sigmoid(x: f64) -> f64 {
    let exp = x.exp();
    exp / (exp + 1.0)
}

pub fn sigmoid_derivative(x: f64) -> f64 {
    let sigmoid = sigmoid(x);
    (1.0 - sigmoid) * sigmoid
}
