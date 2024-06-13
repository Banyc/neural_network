use std::sync::Arc;

use crate::{
    computation::{NodeBackpropagationComputation, NodeComputation, NodeScalarComputation},
    mut_cell::MutCell,
    node::{Node, SharedNode},
    param::SharedParams,
};

pub fn default_bias() -> f64 {
    0.0
}

/// ```math
/// f_b (x) = x + b
/// ```
pub fn bias_node(operand: SharedNode, bias: Option<SharedParams>) -> Node {
    let computation = BiasNodeComputation {};
    let bias = bias.unwrap_or(Arc::new(MutCell::new(vec![default_bias()])));
    assert_eq!(bias.borrow().len(), 1);
    Node::new(
        vec![operand],
        Arc::new(MutCell::new(NodeComputation::Scalar(Box::new(computation)))),
        bias,
    )
}

#[derive(Debug)]
struct BiasNodeComputation {}
impl NodeScalarComputation for BiasNodeComputation {
    fn compute_output(
        &self,
        parameters: &[f64],
        operand_outputs: &[f64],
        _graph_inputs: &[f64],
    ) -> f64 {
        assert_eq!(operand_outputs.len(), 1);
        assert_eq!(parameters.len(), 1);
        bias(operand_outputs[0], parameters[0])
    }
}
impl NodeBackpropagationComputation for BiasNodeComputation {
    fn compute_gradient_of_this_at_operand(
        &self,
        parameters: &[f64],
        operand_outputs: &[f64],
        mut buf: Vec<f64>,
    ) -> Vec<f64> {
        assert_eq!(operand_outputs.len(), 1);
        assert_eq!(parameters.len(), 1);
        buf.extend([bias_derivative()]);
        buf
    }

    fn compute_gradient_of_this_at_parameter(
        &self,
        parameters: &[f64],
        operand_outputs: &[f64],
        mut buf: Vec<f64>,
    ) -> Vec<f64> {
        assert_eq!(operand_outputs.len(), 1);
        assert_eq!(parameters.len(), 1);
        buf.extend([1.0]);
        buf
    }
}

fn bias(x: f64, b: f64) -> f64 {
    x + b
}

fn bias_derivative() -> f64 {
    1.0
}
