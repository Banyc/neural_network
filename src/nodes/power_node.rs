use std::sync::{Arc, Mutex};

use super::node::{Node, NodeComputation};

/// ```math
/// f(x) = x^a
/// ```
pub fn power_node(operand: Arc<Mutex<Node>>, power: f64) -> Node {
    let computation = PowerNodeComputation { power };
    Node::new(vec![operand], Arc::new(computation), Vec::new())
}

#[derive(Debug)]
struct PowerNodeComputation {
    power: f64,
}
impl NodeComputation for PowerNodeComputation {
    fn compute_output(
        &self,
        parameters: &[f64],
        operand_outputs: &[f64],
        _graph_inputs: &[f64],
    ) -> f64 {
        assert!(parameters.is_empty());
        assert_eq!(operand_outputs.len(), 1);
        power(operand_outputs[0], self.power)
    }

    fn compute_gradient_of_this_at_operand(
        &self,
        parameters: &[f64],
        operand_outputs: &[f64],
    ) -> Vec<f64> {
        assert!(parameters.is_empty());
        assert_eq!(operand_outputs.len(), 1);
        vec![power_derivative(operand_outputs[0], self.power)]
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

fn power(base: f64, power: f64) -> f64 {
    base.powf(power)
}

fn power_derivative(base: f64, power: f64) -> f64 {
    power * base.powf(power - 1.)
}
