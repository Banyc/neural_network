use std::sync::{Arc, Mutex};

use super::node::{GeneralNode, NodeComputation};

pub fn l2_error_node(operand: Arc<Mutex<GeneralNode>>, label_index: usize) -> GeneralNode {
    let computation = L2ErrorNodeComputation {
        label_index,
        label: None,
    };
    let node = GeneralNode::new(vec![operand], Box::new(computation), Vec::new());
    node
}

struct L2ErrorNodeComputation {
    label_index: usize,
    label: Option<f64>,
}

impl NodeComputation for L2ErrorNodeComputation {
    fn compute_output(
        &mut self,
        _parameters: &Vec<f64>,
        operand_outputs: &Vec<f64>,
        inputs: &Vec<f64>,
    ) -> f64 {
        self.label = Some(inputs[self.label_index]);
        assert_eq!(operand_outputs.len(), 1);
        l2_error(operand_outputs[0], self.label.unwrap())
    }

    fn compute_local_operand_gradient(
        &self,
        _parameters: &Vec<f64>,
        operand_outputs: &Vec<f64>,
    ) -> Vec<f64> {
        assert_eq!(operand_outputs.len(), 1);
        vec![l2_error_derivative(operand_outputs[0], self.label.unwrap())]
    }

    fn compute_local_parameter_gradient(
        &self,
        _parameters: &Vec<f64>,
        _operand_outputs: &Vec<f64>,
    ) -> Vec<f64> {
        Vec::new()
    }

    fn reset(&mut self) {
        self.label = None;
    }
}

fn l2_error(x: f64, l: f64) -> f64 {
    (x - l).powi(2)
}

fn l2_error_derivative(x: f64, l: f64) -> f64 {
    2.0 * (x - l)
}
