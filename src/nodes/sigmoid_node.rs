use std::sync::{Arc, Mutex};

use super::node::{GeneralNode, NodeComputation};

pub fn sigmoid_node(operand: Arc<Mutex<GeneralNode>>) -> GeneralNode {
    let computation = SigmoidNodeComputation {};
    let node = GeneralNode::new(vec![operand], Box::new(computation), Vec::new());
    node
}

struct SigmoidNodeComputation {}

impl NodeComputation for SigmoidNodeComputation {
    fn compute_output(
        &mut self,
        _parameters: &Vec<f64>,
        operand_outputs: &Vec<f64>,
        _inputs: &Vec<f64>,
    ) -> f64 {
        assert_eq!(operand_outputs.len(), 1);
        sigmoid(operand_outputs[0])
    }

    fn compute_local_operand_gradient(
        &self,
        _parameters: &Vec<f64>,
        operand_outputs: &Vec<f64>,
    ) -> Vec<f64> {
        assert_eq!(operand_outputs.len(), 1);
        vec![sigmoid_derivative(operand_outputs[0])]
    }

    fn compute_local_parameter_gradient(
        &self,
        _parameters: &Vec<f64>,
        _operand_outputs: &Vec<f64>,
    ) -> Vec<f64> {
        Vec::new()
    }

    fn reset(&mut self) {}
}

fn sigmoid(x: f64) -> f64 {
    let exp = x.exp();
    exp / (exp + 1.0)
}

fn sigmoid_derivative(x: f64) -> f64 {
    let sigmoid = sigmoid(x);
    (1.0 - sigmoid) * sigmoid
}
