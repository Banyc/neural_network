use std::sync::{Arc, Mutex};

use crate::node::{Node, NodeComputation};

/// ```math
/// f(x) = \sum x
/// ```
pub fn sum_node(operands: Vec<Arc<Mutex<Node>>>) -> Node {
    let computation = SumNodeComputation {};
    Node::new(operands, Arc::new(computation), Vec::new())
}

#[derive(Debug)]
struct SumNodeComputation {}
impl NodeComputation for SumNodeComputation {
    fn compute_output(
        &self,
        parameters: &[f64],
        operand_outputs: &[f64],
        _graph_inputs: &[f64],
    ) -> f64 {
        assert!(parameters.is_empty());
        sum(operand_outputs)
    }

    fn compute_gradient_of_this_at_operand(
        &self,
        parameters: &[f64],
        operand_outputs: &[f64],
    ) -> Vec<f64> {
        assert!(parameters.is_empty());
        sum_derivative(operand_outputs)
    }

    fn compute_gradient_of_this_at_parameter(
        &self,
        parameters: &[f64],
        operand_outputs: &[f64],
    ) -> Vec<f64> {
        assert!(parameters.is_empty());
        assert_eq!(operand_outputs.len(), 1);
        vec![]
    }
}

fn sum(x: &[f64]) -> f64 {
    x.iter().sum()
}

fn sum_derivative(x: &[f64]) -> Vec<f64> {
    x.iter().map(|_| 1.).collect()
}
