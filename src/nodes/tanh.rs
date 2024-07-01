use graph::NodeIdx;

use crate::{
    computation::{NodeBackpropagationComputation, NodeComputation, NodeScalarComputation},
    mut_cell::MutCell,
    node::CompNode,
    param::empty_shared_params,
    ref_ctr::RefCtr,
};

/// ```math
/// f(x) = \frac{e^{2x} - 1}{e^{2x} + 1}
/// ```
pub fn tanh_node(operand: NodeIdx) -> CompNode {
    let computation = SigmoidNodeComputation {};
    CompNode::new(
        vec![operand],
        RefCtr::new(MutCell::new(NodeComputation::Scalar(Box::new(computation)))),
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
        tanh(operand_outputs[0])
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
        buf.extend([tanh_derivative(operand_outputs[0])]);
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

pub fn tanh(x: f64) -> f64 {
    // Prevent floating point underflow
    let x = x.clamp(-250.0, 250.0);
    let e2x = f64::exp(2. * x);
    (e2x - 1.0) / (e2x + 1.0)
}

pub fn tanh_derivative(x: f64) -> f64 {
    let tanh2 = tanh(x).powi(2);
    1. - tanh2
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_underflow() {
        let x = f64::MIN;
        let exp = f64::exp(-x);
        assert!(exp.is_infinite());

        let s = tanh(x);
        assert!(s.is_finite());
        assert!(s < 0.001);
    }
}
