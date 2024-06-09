use std::sync::{Arc, Mutex};

use strict_num::FiniteF64;

use crate::{
    node::{Node, NodeComputation},
    param::empty_shared_params,
};

/// ```math
/// f(x) = \max x
/// ```
pub fn max_node(operands: Vec<Arc<Mutex<Node>>>) -> Node {
    assert!(!operands.is_empty());
    let computation = MaxNodeComputation {};
    Node::new(operands, Arc::new(computation), empty_shared_params())
}

#[derive(Debug)]
struct MaxNodeComputation {}
impl NodeComputation for MaxNodeComputation {
    fn compute_output(
        &self,
        parameters: &[f64],
        operand_outputs: &[f64],
        _graph_inputs: &[f64],
    ) -> f64 {
        assert!(parameters.is_empty());
        assert!(!operand_outputs.is_empty());
        max(operand_outputs)
    }

    fn compute_gradient_of_this_at_operand(
        &self,
        parameters: &[f64],
        operand_outputs: &[f64],
    ) -> Vec<f64> {
        assert!(parameters.is_empty());
        assert!(!operand_outputs.is_empty());
        max_derivative(operand_outputs)
    }

    fn compute_gradient_of_this_at_parameter(
        &self,
        parameters: &[f64],
        operand_outputs: &[f64],
    ) -> Vec<f64> {
        assert!(parameters.is_empty());
        assert!(!operand_outputs.is_empty());
        vec![]
    }
}

fn max(x: &[f64]) -> f64 {
    x.iter()
        .copied()
        .map(|x| FiniteF64::new(x).unwrap())
        .max()
        .unwrap()
        .get()
}

fn max_derivative(x: &[f64]) -> Vec<f64> {
    let max = max(x);
    let i = x.iter().copied().position(|x| x == max).unwrap();
    (0..x.len()).map(|j| if i == j { 1. } else { 0. }).collect()
}
