use std::sync::{Arc, Mutex};

use crate::node::{Node, NodeComputation};

/// ```math
/// f(x) = \frac{e^{x_i}}{\sum e^x}
/// ```
pub fn softmax_node(operand: Arc<Mutex<Node>>, operand_index: usize) -> Node {
    let computation = SoftmaxNodeComputation { operand_index };
    Node::new(vec![operand], Arc::new(computation), Vec::new())
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
        assert!(0 < operand_outputs.len());
        softmax(operand_outputs, self.operand_index)
    }

    fn compute_gradient_of_this_at_operand(
        &self,
        parameters: &[f64],
        operand_outputs: &[f64],
    ) -> Vec<f64> {
        assert!(parameters.is_empty());
        assert!(0 < operand_outputs.len());
        softmax_derivative(operand_outputs, self.operand_index)
    }

    fn compute_gradient_of_this_at_parameter(
        &self,
        parameters: &[f64],
        operand_outputs: &[f64],
    ) -> Vec<f64> {
        assert!(parameters.is_empty());
        assert!(0 < operand_outputs.len());
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
