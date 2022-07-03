use std::sync::{Arc, Mutex};

use super::node::{GeneralNode, NodeComputation};

pub fn l2_error_node(
    operand: Arc<Mutex<GeneralNode>>,
    label: Arc<Mutex<GeneralNode>>,
) -> GeneralNode {
    let computation = L2ErrorNodeComputation {};
    let node = GeneralNode::new(vec![operand, label], Box::new(computation), Vec::new());
    node
}

struct L2ErrorNodeComputation {}

impl NodeComputation for L2ErrorNodeComputation {
    fn compute_output(
        &self,
        _parameters: &Vec<f64>,
        operand_outputs: &Vec<f64>,
        _inputs: &Vec<f64>,
    ) -> f64 {
        assert_eq!(operand_outputs.len(), 2);
        l2_error(operand_outputs[0], operand_outputs[1])
    }

    fn compute_local_operand_gradient(
        &self,
        _parameters: &Vec<f64>,
        operand_outputs: &Vec<f64>,
    ) -> Vec<f64> {
        assert_eq!(operand_outputs.len(), 2);
        vec![
            l2_error_derivative(operand_outputs[0], operand_outputs[1]),
            -l2_error_derivative(operand_outputs[0], operand_outputs[1]),
        ]
    }

    fn compute_local_parameter_gradient(
        &self,
        _parameters: &Vec<f64>,
        _operand_outputs: &Vec<f64>,
    ) -> Vec<f64> {
        Vec::new()
    }
}

fn l2_error(x: f64, l: f64) -> f64 {
    (x - l).powi(2)
}

fn l2_error_derivative(x: f64, l: f64) -> f64 {
    2.0 * (x - l)
}
