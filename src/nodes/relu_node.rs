use std::sync::{Arc, Mutex};

use crate::node::{Node, NodeComputation};

/// ```math
/// f(x) = \begin{cases}
///   x & x \geq 0 \\
///   0 & x < 0 \\
/// \end{cases}
/// ```
pub fn relu_node(operand: Arc<Mutex<Node>>) -> Node {
    let computation = ReluNodeComputation {};
    Node::new(vec![operand], Arc::new(computation), Vec::new())
}

#[derive(Debug)]
struct ReluNodeComputation {}
impl NodeComputation for ReluNodeComputation {
    fn compute_output(
        &self,
        parameters: &[f64],
        operand_outputs: &[f64],
        _graph_inputs: &[f64],
    ) -> f64 {
        assert!(parameters.is_empty());
        assert_eq!(operand_outputs.len(), 1);
        relu(operand_outputs[0])
    }

    fn compute_gradient_of_this_at_operand(
        &self,
        parameters: &[f64],
        operand_outputs: &[f64],
    ) -> Vec<f64> {
        assert!(parameters.is_empty());
        assert_eq!(operand_outputs.len(), 1);
        vec![relu_derivative(operand_outputs[0])]
    }

    fn compute_gradient_of_this_at_parameter(
        &self,
        parameters: &[f64],
        operand_outputs: &[f64],
    ) -> Vec<f64> {
        assert!(parameters.is_empty());
        assert_eq!(operand_outputs.len(), 1);
        Vec::new()
    }
}

fn relu(x: f64) -> f64 {
    f64::max(x, 0.0)
}

fn relu_derivative(x: f64) -> f64 {
    match x {
        _ if x > 0.0 => 1.0,
        _ => 0.0,
    }
}

#[cfg(test)]
mod tests {
    use crate::nodes::input_node::input_node;

    use super::*;

    #[test]
    fn evaluate_negative() {
        let input_node = input_node(0);
        let mut relu = relu_node(Arc::new(Mutex::new(input_node)));
        let ret = relu.evaluate_once(&[-2.0]);
        assert!(ret >= 0.0);
        assert!(ret <= 0.0);
    }

    #[test]
    fn evaluate_positive() {
        let input_node = input_node(0);
        let mut relu = relu_node(Arc::new(Mutex::new(input_node)));
        let ret = relu.evaluate_once(&[3.0]);
        assert!(ret >= 3.0);
        assert!(ret <= 3.0);
    }

    #[test]
    fn positive_gradient_of_this_at_operand() {
        let input_node = input_node(0);
        let mut relu = relu_node(Arc::new(Mutex::new(input_node)));
        relu.evaluate_once(&[3.0]);
        let ret = relu.gradient_of_this_at_operand().unwrap();
        assert!(ret[0] >= 1.0);
        assert!(ret[0] <= 1.0);
    }

    #[test]
    fn negative_gradient_of_this_at_operand() {
        let input_node = input_node(0);
        let mut relu = relu_node(Arc::new(Mutex::new(input_node)));
        relu.evaluate_once(&[-3.0]);
        let ret = relu.gradient_of_this_at_operand().unwrap();
        assert!(ret[0] >= 0.0);
        assert!(ret[0] <= 0.0);
    }

    #[test]
    fn empty_gradient_of_this_at_parameter() {
        let input_node = input_node(0);
        let mut relu = relu_node(Arc::new(Mutex::new(input_node)));
        relu.evaluate_once(&[3.0]);
        let ret = relu.gradient_of_this_at_parameter().unwrap();
        assert_eq!(ret.len(), 0);
    }
}
