use std::sync::Arc;

use crate::{
    node::{Node, NodeComputation, SharedNode},
    param::empty_shared_params,
};

/// ```math
/// f(x, l) = (x - l)^2
/// ```
pub fn l2_error_node(operand: SharedNode, label: SharedNode) -> Node {
    let computation = L2ErrorNodeComputation {};
    Node::new(
        vec![operand, label],
        Arc::new(computation),
        empty_shared_params(),
    )
}

#[derive(Debug)]
struct L2ErrorNodeComputation {}
impl NodeComputation for L2ErrorNodeComputation {
    fn compute_output(
        &self,
        parameters: &[f64],
        operand_outputs: &[f64],
        _graph_inputs: &[f64],
    ) -> f64 {
        assert!(parameters.is_empty());
        assert_eq!(operand_outputs.len(), 2);
        l2_error(operand_outputs[0], operand_outputs[1])
    }

    fn compute_gradient_of_this_at_operand(
        &self,
        parameters: &[f64],
        operand_outputs: &[f64],
        mut buf: Vec<f64>,
    ) -> Vec<f64> {
        assert!(parameters.is_empty());
        assert_eq!(operand_outputs.len(), 2);
        buf.extend([
            l2_error_derivative(operand_outputs[0], operand_outputs[1]),
            -l2_error_derivative(operand_outputs[0], operand_outputs[1]),
        ]);
        buf
    }

    fn compute_gradient_of_this_at_parameter(
        &self,
        parameters: &[f64],
        operand_outputs: &[f64],
        buf: Vec<f64>,
    ) -> Vec<f64> {
        assert!(parameters.is_empty());
        assert_eq!(operand_outputs.len(), 2);
        buf
    }
}

fn l2_error(x: f64, l: f64) -> f64 {
    (x - l).powi(2)
}

fn l2_error_derivative(x: f64, l: f64) -> f64 {
    2.0 * (x - l)
}
