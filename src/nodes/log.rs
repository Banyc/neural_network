use crate::{
    computation::{NodeBackpropagationComputation, NodeComputation, NodeScalarComputation},
    mut_cell::MutCell,
    node::{Node, SharedNode},
    param::empty_shared_params,
    ref_ctr::RefCtr,
};

/// ```math
/// f(x) = \log_a x
/// ```
pub fn log_node(operand: SharedNode, base: f64) -> Node {
    let computation = LogNodeComputation { base };
    Node::new(
        vec![operand],
        RefCtr::new(MutCell::new(NodeComputation::Scalar(Box::new(computation)))),
        empty_shared_params(),
    )
}

#[derive(Debug)]
struct LogNodeComputation {
    base: f64,
}
impl NodeScalarComputation for LogNodeComputation {
    fn compute_output(
        &self,
        parameters: &[f64],
        operand_outputs: &[f64],
        _graph_inputs: &[f64],
    ) -> f64 {
        assert!(parameters.is_empty());
        assert_eq!(operand_outputs.len(), 1);
        log(operand_outputs[0], self.base)
    }
}
impl NodeBackpropagationComputation for LogNodeComputation {
    fn compute_gradient_of_this_at_operand(
        &self,
        parameters: &[f64],
        operand_outputs: &[f64],
        mut buf: Vec<f64>,
    ) -> Vec<f64> {
        assert!(parameters.is_empty());
        assert_eq!(operand_outputs.len(), 1);
        buf.extend([log_derivative(operand_outputs[0], self.base)]);
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

fn log(x: f64, base: f64) -> f64 {
    x.log(base)
}

fn log_derivative(x: f64, base: f64) -> f64 {
    1. / (x * base.ln())
}
