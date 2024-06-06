use std::sync::{Arc, Mutex};

use super::node::{Node, NodeComputation};

/// ```math
/// f_b (x) = x + b
/// ```
pub fn bias_node(operand: Arc<Mutex<Node>>, bias: Option<f64>) -> Node {
    let computation = BiasNodeComputation {};
    let bias = bias.unwrap_or(0.0);
    Node::new(vec![operand], Arc::new(computation), vec![bias])
}

#[derive(Debug)]
struct BiasNodeComputation {}
impl NodeComputation for BiasNodeComputation {
    fn compute_output(
        &self,
        parameters: &[f64],
        operand_outputs: &[f64],
        _graph_inputs: &[f64],
    ) -> f64 {
        assert_eq!(operand_outputs.len(), 1);
        assert_eq!(parameters.len(), 1);
        bias(operand_outputs[0], parameters[0])
    }

    fn compute_gradient_of_this_at_operand(
        &self,
        parameters: &[f64],
        operand_outputs: &[f64],
    ) -> Vec<f64> {
        assert_eq!(operand_outputs.len(), 1);
        assert_eq!(parameters.len(), 1);
        vec![bias_derivative()]
    }

    fn compute_gradient_of_this_at_parameter(
        &self,
        parameters: &[f64],
        operand_outputs: &[f64],
    ) -> Vec<f64> {
        assert_eq!(operand_outputs.len(), 1);
        assert_eq!(parameters.len(), 1);
        vec![1.0]
    }
}

fn bias(x: f64, b: f64) -> f64 {
    x + b
}

fn bias_derivative() -> f64 {
    1.0
}
