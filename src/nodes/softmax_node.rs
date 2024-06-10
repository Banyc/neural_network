use std::sync::Arc;

use crate::{
    node::{Node, NodeComputation, SharedNode},
    param::empty_shared_params,
};

/// ```math
/// f(x) = \frac{e^{x_i}}{\sum e^x}
/// ```
pub fn softmax_node(operand: SharedNode, operand_index: usize) -> Node {
    let computation = SoftmaxNodeComputation { operand_index };
    Node::new(vec![operand], Arc::new(computation), empty_shared_params())
}

#[derive(Debug)]
struct SoftmaxNodeComputation {
    operand_index: usize,
}
impl NodeComputation for SoftmaxNodeComputation {
    fn compute_output(
        &self,
        parameters: &[f64],
        operand_outputs: &[f64],
        _graph_inputs: &[f64],
    ) -> f64 {
        assert!(parameters.is_empty());
        assert!(!operand_outputs.is_empty());
        softmax(operand_outputs, self.operand_index)
    }

    fn compute_gradient_of_this_at_operand(
        &self,
        parameters: &[f64],
        operand_outputs: &[f64],
    ) -> Vec<f64> {
        assert!(parameters.is_empty());
        assert!(!operand_outputs.is_empty());
        softmax_derivative(operand_outputs, self.operand_index)
    }

    fn compute_gradient_of_this_at_parameter(
        &self,
        parameters: &[f64],
        operand_outputs: &[f64],
    ) -> Vec<f64> {
        assert!(parameters.is_empty());
        assert!(!operand_outputs.is_empty());
        Vec::new()
    }
}

fn softmax(x: &[f64], i: usize) -> f64 {
    let e_x_i = x[i].exp();
    let e_x = x.iter().copied().map(|x| x.exp());
    e_x_i / e_x.into_iter().sum::<f64>()
}

fn softmax_derivative(x: &[f64], i: usize) -> Vec<f64> {
    let p_i = softmax(x, i);
    let mut der = vec![];
    for j in 0..x.len() {
        let d = if i == j {
            p_i * (1. - p_i)
        } else {
            let p_j = softmax(x, j);
            -p_i * p_j
        };
        der.push(d);
    }
    der
}
