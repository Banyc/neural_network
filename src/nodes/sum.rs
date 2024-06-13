use std::sync::Arc;

use crate::{
    computation::{NodeBackpropagationComputation, NodeComputation, NodeScalarComputation},
    mut_cell::MutCell,
    node::{Node, SharedNode},
    param::empty_shared_params,
};

/// ```math
/// f(x) = \sum x
/// ```
pub fn sum_node(operands: Vec<SharedNode>) -> Node {
    let computation = SumNodeComputation {};
    Node::new(
        operands,
        Arc::new(MutCell::new(NodeComputation::Scalar(Box::new(computation)))),
        empty_shared_params(),
    )
}

#[derive(Debug)]
struct SumNodeComputation {}
impl NodeScalarComputation for SumNodeComputation {
    fn compute_output(
        &self,
        parameters: &[f64],
        operand_outputs: &[f64],
        _graph_inputs: &[f64],
    ) -> f64 {
        assert!(parameters.is_empty());
        sum(operand_outputs)
    }
}
impl NodeBackpropagationComputation for SumNodeComputation {
    fn compute_gradient_of_this_at_operand(
        &self,
        parameters: &[f64],
        operand_outputs: &[f64],
        buf: Vec<f64>,
    ) -> Vec<f64> {
        assert!(parameters.is_empty());
        sum_derivative(operand_outputs, buf)
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

fn sum(x: &[f64]) -> f64 {
    x.iter().sum()
}

fn sum_derivative(x: &[f64], mut buf: Vec<f64>) -> Vec<f64> {
    buf.extend(x.iter().map(|_| 1.));
    buf
}
