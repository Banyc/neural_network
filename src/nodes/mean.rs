use crate::{
    computation::{NodeBackpropagationComputation, NodeComputation, NodeScalarComputation},
    mut_cell::MutCell,
    node::{Node, SharedNode},
    param::empty_shared_params,
    ref_ctr::RefCtr,
};

/// ```math
/// f(x) = \frac{\sum x}{n}
/// ```
pub fn mean_node(operands: Vec<SharedNode>) -> Node {
    let computation = MeanNodeComputation {};
    Node::new(
        operands,
        RefCtr::new(MutCell::new(NodeComputation::Scalar(Box::new(computation)))),
        empty_shared_params(),
    )
}

#[derive(Debug)]
struct MeanNodeComputation {}
impl NodeScalarComputation for MeanNodeComputation {
    fn compute_output(
        &self,
        parameters: &[f64],
        operand_outputs: &[f64],
        _graph_inputs: &[f64],
    ) -> f64 {
        assert!(parameters.is_empty());
        assert!(!operand_outputs.is_empty());
        mean(operand_outputs)
    }
}
impl NodeBackpropagationComputation for MeanNodeComputation {
    fn compute_gradient_of_this_at_operand(
        &self,
        parameters: &[f64],
        operand_outputs: &[f64],
        buf: Vec<f64>,
    ) -> Vec<f64> {
        assert!(parameters.is_empty());
        assert!(!operand_outputs.is_empty());
        mean_derivative(operand_outputs, buf)
    }

    fn compute_gradient_of_this_at_parameter(
        &self,
        parameters: &[f64],
        operand_outputs: &[f64],
        buf: Vec<f64>,
    ) -> Vec<f64> {
        assert!(parameters.is_empty());
        assert!(!operand_outputs.is_empty());
        buf
    }
}

fn mean(x: &[f64]) -> f64 {
    let n = x.len();
    x.iter().copied().map(|x| x / n as f64).sum::<f64>()
}

fn mean_derivative(x: &[f64], mut buf: Vec<f64>) -> Vec<f64> {
    let n = x.len();
    buf.extend(core::iter::repeat(1. / n as f64).take(n));
    buf
}
