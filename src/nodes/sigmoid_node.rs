use std::{cell::RefCell, rc::Rc};

use super::node::{GeneralNode, NodeComputation};

pub fn sigmoid_node(operand: Rc<RefCell<GeneralNode>>) -> GeneralNode {
    let computation = SigmoidNodeComputation {};
    GeneralNode::new(vec![operand], Box::new(computation), Vec::new())
}

struct SigmoidNodeComputation {}

impl NodeComputation for SigmoidNodeComputation {
    fn compute_output(&self, _parameters: &[f64], operand_outputs: &[f64], _inputs: &[f64]) -> f64 {
        assert_eq!(operand_outputs.len(), 1);
        sigmoid(operand_outputs[0])
    }

    fn compute_gradient_of_this_at_operand(
        &self,
        _parameters: &[f64],
        operand_outputs: &[f64],
    ) -> Vec<f64> {
        assert_eq!(operand_outputs.len(), 1);
        vec![sigmoid_derivative(operand_outputs[0])]
    }

    fn compute_gradient_of_this_at_parameter(
        &self,
        _parameters: &[f64],
        _operand_outputs: &[f64],
    ) -> Vec<f64> {
        Vec::new()
    }
}

fn sigmoid(x: f64) -> f64 {
    let exp = x.exp();
    exp / (exp + 1.0)
}

fn sigmoid_derivative(x: f64) -> f64 {
    let sigmoid = sigmoid(x);
    (1.0 - sigmoid) * sigmoid
}
