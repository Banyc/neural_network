use std::sync::{Arc, Mutex};

use super::node::{Node, NodeComputation};

pub fn log_node(operand: Arc<Mutex<Node>>, base: f64) -> Node {
    let computation = LogNodeComputation { base };
    Node::new(vec![operand], Arc::new(computation), Vec::new())
}

#[derive(Debug)]
struct LogNodeComputation {
    base: f64,
}
impl NodeComputation for LogNodeComputation {
    fn compute_output(&self, parameters: &[f64], operand_outputs: &[f64], _inputs: &[f64]) -> f64 {
        assert!(parameters.is_empty());
        assert_eq!(operand_outputs.len(), 1);
        log(operand_outputs[0], self.base)
    }

    fn compute_gradient_of_this_at_operand(
        &self,
        parameters: &[f64],
        operand_outputs: &[f64],
    ) -> Vec<f64> {
        assert!(parameters.is_empty());
        assert_eq!(operand_outputs.len(), 1);
        vec![log_derivative(operand_outputs[0], self.base)]
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

fn log(x: f64, base: f64) -> f64 {
    x.log(base)
}

fn log_derivative(x: f64, base: f64) -> f64 {
    1. / (x * base.ln())
}
