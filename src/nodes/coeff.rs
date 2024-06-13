use std::sync::Arc;

use crate::{
    computation::{NodeBackpropagationComputation, NodeComputation, NodeScalarComputation},
    mut_cell::MutCell,
    node::{Node, SharedNode},
    param::empty_shared_params,
};

/// ```math
/// f(x) = ax
/// ```
pub fn coeff_node(operand: SharedNode, coefficient: f64) -> Node {
    let computation = CoeffNodeComputation { coefficient };
    Node::new(
        vec![operand],
        Arc::new(MutCell::new(NodeComputation::Scalar(Box::new(computation)))),
        empty_shared_params(),
    )
}

#[derive(Debug)]
struct CoeffNodeComputation {
    coefficient: f64,
}
impl NodeScalarComputation for CoeffNodeComputation {
    fn compute_output(
        &self,
        parameters: &[f64],
        operand_outputs: &[f64],
        _graph_inputs: &[f64],
    ) -> f64 {
        assert!(parameters.is_empty());
        assert_eq!(operand_outputs.len(), 1);
        self.coefficient * operand_outputs[0]
    }
}
impl NodeBackpropagationComputation for CoeffNodeComputation {
    fn compute_gradient_of_this_at_operand(
        &self,
        parameters: &[f64],
        operand_outputs: &[f64],
        mut buf: Vec<f64>,
    ) -> Vec<f64> {
        assert!(parameters.is_empty());
        assert_eq!(operand_outputs.len(), 1);
        buf.extend([self.coefficient]);
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
