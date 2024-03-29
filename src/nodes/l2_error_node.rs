use std::{cell::RefCell, rc::Rc};

use super::node::{GeneralNode, NodeComputation};

pub fn l2_error_node(
    operand: Rc<RefCell<GeneralNode>>,
    label: Rc<RefCell<GeneralNode>>,
) -> GeneralNode {
    let computation = L2ErrorNodeComputation {};
    GeneralNode::new(vec![operand, label], Box::new(computation), Vec::new())
}

struct L2ErrorNodeComputation {}

impl NodeComputation for L2ErrorNodeComputation {
    fn compute_output(&self, _parameters: &[f64], operand_outputs: &[f64], _inputs: &[f64]) -> f64 {
        assert_eq!(operand_outputs.len(), 2);
        l2_error(operand_outputs[0], operand_outputs[1])
    }

    fn compute_gradient_of_this_at_operand(
        &self,
        _parameters: &[f64],
        operand_outputs: &[f64],
    ) -> Vec<f64> {
        assert_eq!(operand_outputs.len(), 2);
        vec![
            l2_error_derivative(operand_outputs[0], operand_outputs[1]),
            -l2_error_derivative(operand_outputs[0], operand_outputs[1]),
        ]
    }

    fn compute_gradient_of_this_at_parameter(
        &self,
        _parameters: &[f64],
        _operand_outputs: &[f64],
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
