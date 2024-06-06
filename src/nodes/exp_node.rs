use std::sync::{Arc, Mutex};

use super::node::{Node, NodeComputation};

/// ```math
/// f(x) = a^x
/// ```
pub fn exp_node(operand: Arc<Mutex<Node>>, base: f64) -> Node {
    let computation = ExpNodeComputation { base };
    Node::new(vec![operand], Arc::new(computation), Vec::new())
}

#[derive(Debug)]
struct ExpNodeComputation {
    base: f64,
}
impl NodeComputation for ExpNodeComputation {
    fn compute_output(&self, parameters: &[f64], operand_outputs: &[f64], _inputs: &[f64]) -> f64 {
        assert!(parameters.is_empty());
        assert_eq!(operand_outputs.len(), 1);
        exp(self.base, operand_outputs[0])
    }

    fn compute_gradient_of_this_at_operand(
        &self,
        parameters: &[f64],
        operand_outputs: &[f64],
    ) -> Vec<f64> {
        assert!(parameters.is_empty());
        assert_eq!(operand_outputs.len(), 1);
        vec![exp_derivative(self.base, operand_outputs[0])]
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

fn exp(base: f64, power: f64) -> f64 {
    base.powf(power)
}

fn exp_derivative(base: f64, power: f64) -> f64 {
    exp(base, power) * base.ln()
}
