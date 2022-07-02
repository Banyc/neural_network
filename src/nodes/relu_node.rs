use std::sync::{Arc, Mutex};

use super::node::{GeneralNode, NodeComputation};

pub fn relu_node(operand: Arc<Mutex<GeneralNode>>) -> GeneralNode {
    let computation = ReluNodeComputation {};
    let node = GeneralNode::new(vec![operand], Box::new(computation), Vec::new());
    node
}

struct ReluNodeComputation {}

impl NodeComputation for ReluNodeComputation {
    fn compute_output(
        &mut self,
        _parameters: &Vec<f64>,
        operand_outputs: &Vec<f64>,
        _inputs: &Vec<f64>,
    ) -> f64 {
        assert_eq!(operand_outputs.len(), 1);
        relu(operand_outputs[0])
    }

    fn compute_local_operand_gradient(
        &self,
        _parameters: &Vec<f64>,
        operand_outputs: &Vec<f64>,
    ) -> Vec<f64> {
        assert_eq!(operand_outputs.len(), 1);
        vec![relu_derivative(operand_outputs[0])]
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

fn relu(x: f64) -> f64 {
    f64::max(x, 0.0)
}

fn relu_derivative(x: f64) -> f64 {
    match x {
        _ if x > 0.0 => 1.0,
        _ => 0.0,
    }
}
