use std::sync::Arc;

use crate::{
    computation::{NodeBackpropagationComputation, NodeComputation, NodeScalarComputation},
    mut_cell::MutCell,
    node::{Node, SharedNode},
    param::empty_shared_params,
};

pub fn softmax_layer(operands: Vec<SharedNode>) -> Vec<SharedNode> {
    let mut layer = vec![];
    for i in 0..operands.len() {
        let node = Arc::new(MutCell::new(softmax_node(operands.clone(), i)));
        layer.push(node);
    }
    layer
}

/// ```math
/// f(x) = \frac{e^{x_i}}{\sum e^x}
/// ```
pub fn softmax_node(operands: Vec<SharedNode>, operand_index: usize) -> Node {
    let computation = SoftmaxNodeComputation { operand_index };
    Node::new(
        operands,
        Arc::new(MutCell::new(NodeComputation::Scalar(Box::new(computation)))),
        empty_shared_params(),
    )
}

#[derive(Debug)]
struct SoftmaxNodeComputation {
    operand_index: usize,
}
impl NodeScalarComputation for SoftmaxNodeComputation {
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
}
impl NodeBackpropagationComputation for SoftmaxNodeComputation {
    fn compute_gradient_of_this_at_operand(
        &self,
        parameters: &[f64],
        operand_outputs: &[f64],
        buf: Vec<f64>,
    ) -> Vec<f64> {
        assert!(parameters.is_empty());
        assert!(!operand_outputs.is_empty());
        softmax_derivative(operand_outputs, self.operand_index, buf)
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

fn softmax(x: &[f64], i: usize) -> f64 {
    let e_x_i = x[i].exp();
    let e_x = x.iter().copied().map(|x| x.exp());
    e_x_i / e_x.into_iter().sum::<f64>()
}

fn softmax_derivative(x: &[f64], i: usize, mut buf: Vec<f64>) -> Vec<f64> {
    let p_i = softmax(x, i);
    for j in 0..x.len() {
        let d = if i == j {
            p_i * (1. - p_i)
        } else {
            let p_j = softmax(x, j);
            -p_i * p_j
        };
        buf.push(d);
    }
    buf
}
