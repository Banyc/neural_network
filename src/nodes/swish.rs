use std::sync::Arc;

use crate::{
    computation::{NodeBackpropagationComputation, NodeComputation, NodeScalarComputation},
    mut_cell::MutCell,
    node::{Node, SharedNode},
    param::empty_shared_params,
};

use super::sigmoid::sigmoid;

/// ```math
/// f(x) = x \text{sigmoid}(x)
/// ```
pub fn swish_node(operand: SharedNode) -> Node {
    let computation = SwishNodeComputation {};
    Node::new(
        vec![operand],
        Arc::new(MutCell::new(NodeComputation::Scalar(Box::new(computation)))),
        empty_shared_params(),
    )
}

#[derive(Debug)]
struct SwishNodeComputation {}
impl NodeScalarComputation for SwishNodeComputation {
    fn compute_output(
        &self,
        parameters: &[f64],
        operand_outputs: &[f64],
        _graph_inputs: &[f64],
    ) -> f64 {
        assert!(parameters.is_empty());
        assert_eq!(operand_outputs.len(), 1);
        swish(operand_outputs[0])
    }
}
impl NodeBackpropagationComputation for SwishNodeComputation {
    fn compute_gradient_of_this_at_operand(
        &self,
        parameters: &[f64],
        operand_outputs: &[f64],
        mut buf: Vec<f64>,
    ) -> Vec<f64> {
        assert!(parameters.is_empty());
        assert_eq!(operand_outputs.len(), 1);
        buf.extend([swish_derivative(operand_outputs[0])]);
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

fn swish(x: f64) -> f64 {
    x * sigmoid(x)
}

fn swish_derivative(x: f64) -> f64 {
    // sigmoid(x) + x * sigmoid_derivative(x)

    let swish = swish(x);
    swish + sigmoid(x) * (1. - swish)
}
