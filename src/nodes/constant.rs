use std::sync::Arc;

use crate::{
    computation::{NodeBackpropagationComputation, NodeComputation, NodeScalarComputation},
    mut_cell::MutCell,
    node::Node,
    param::empty_shared_params,
};

pub fn constant_node(value: f64) -> Node {
    let computation = ConstantNodeComputation { value };
    Node::new(
        vec![],
        Arc::new(MutCell::new(NodeComputation::Scalar(Box::new(computation)))),
        empty_shared_params(),
    )
}

#[derive(Debug)]
struct ConstantNodeComputation {
    value: f64,
}
impl NodeScalarComputation for ConstantNodeComputation {
    fn compute_output(
        &self,
        parameters: &[f64],
        operand_outputs: &[f64],
        _graph_inputs: &[f64],
    ) -> f64 {
        assert!(parameters.is_empty());
        assert!(operand_outputs.is_empty());
        self.value
    }
}
impl NodeBackpropagationComputation for ConstantNodeComputation {
    fn compute_gradient_of_this_at_operand(
        &self,
        parameters: &[f64],
        operand_outputs: &[f64],
        buf: Vec<f64>,
    ) -> Vec<f64> {
        assert!(parameters.is_empty());
        assert!(operand_outputs.is_empty());
        buf
    }

    fn compute_gradient_of_this_at_parameter(
        &self,
        parameters: &[f64],
        operand_outputs: &[f64],
        buf: Vec<f64>,
    ) -> Vec<f64> {
        assert!(parameters.is_empty());
        assert!(operand_outputs.is_empty());
        buf
    }
}
