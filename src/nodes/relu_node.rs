use std::{cell::RefCell, rc::Rc};

use super::node::{GeneralNode, NodeComputation};

pub fn relu_node(operand: Rc<RefCell<GeneralNode>>) -> GeneralNode {
    let computation = ReluNodeComputation {};
    GeneralNode::new(vec![operand], Box::new(computation), Vec::new())
}

struct ReluNodeComputation {}

impl NodeComputation for ReluNodeComputation {
    fn compute_output(&self, _parameters: &[f64], operand_outputs: &[f64], _inputs: &[f64]) -> f64 {
        assert_eq!(operand_outputs.len(), 1);
        relu(operand_outputs[0])
    }

    fn compute_gradient_of_this_at_operand(
        &self,
        _parameters: &[f64],
        operand_outputs: &[f64],
    ) -> Vec<f64> {
        assert_eq!(operand_outputs.len(), 1);
        vec![relu_derivative(operand_outputs[0])]
    }

    fn compute_gradient_of_this_at_parameter(
        &self,
        _parameters: &[f64],
        _operand_outputs: &[f64],
    ) -> Vec<f64> {
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
    use std::{cell::RefCell, rc::Rc};

    use super::super::{input_node::input_node, relu_node::relu_node};

    #[test]
    fn evaluate_negative() {
        let input_node = input_node(0);
        let mut relu = relu_node(Rc::new(RefCell::new(input_node)));
        let ret = relu.evaluate(&[-2.0]);
        assert!(ret >= 0.0);
        assert!(ret <= 0.0);
    }

    #[test]
    fn evaluate_positive() {
        let input_node = input_node(0);
        let mut relu = relu_node(Rc::new(RefCell::new(input_node)));
        let ret = relu.evaluate(&[3.0]);
        assert!(ret >= 3.0);
        assert!(ret <= 3.0);
    }

    #[test]
    fn positive_gradient_of_this_at_operand() {
        let input_node = input_node(0);
        let mut relu = relu_node(Rc::new(RefCell::new(input_node)));
        relu.evaluate(&[3.0]);
        let ret = relu.gradient_of_this_at_operand().unwrap();
        assert!(ret[0] >= 1.0);
        assert!(ret[0] <= 1.0);
    }

    #[test]
    fn negative_gradient_of_this_at_operand() {
        let input_node = input_node(0);
        let mut relu = relu_node(Rc::new(RefCell::new(input_node)));
        relu.evaluate(&[-3.0]);
        let ret = relu.gradient_of_this_at_operand().unwrap();
        assert!(ret[0] >= 0.0);
        assert!(ret[0] <= 0.0);
    }

    #[test]
    fn empty_gradient_of_this_at_parameter() {
        let input_node = input_node(0);
        let mut relu = relu_node(Rc::new(RefCell::new(input_node)));
        relu.evaluate(&[3.0]);
        let ret = relu.gradient_of_this_at_parameter().unwrap();
        assert_eq!(ret.len(), 0);
    }
}
