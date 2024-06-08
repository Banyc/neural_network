use std::sync::{Arc, Mutex};

use crate::node::{Node, NodeComputation};

/// ```math
/// f(y, \hat{y}) = \frac{1}{n} \sum (y - \hat{y})^2
/// ```
pub fn mse_node(operands: Vec<Arc<Mutex<Node>>>) -> Node {
    let computation = MseNodeComputation {};
    Node::new(operands, Arc::new(computation), Vec::new())
}

#[derive(Debug)]
struct MseNodeComputation {}
impl NodeComputation for MseNodeComputation {
    fn compute_output(
        &self,
        parameters: &[f64],
        operand_outputs: &[f64],
        _graph_inputs: &[f64],
    ) -> f64 {
        assert!(parameters.is_empty());
        assert!(!operand_outputs.is_empty());
        let n = operand_outputs.len() / 2;
        let y = &operand_outputs[..n];
        let y_hat = &operand_outputs[n..];
        mse(y, y_hat)
    }

    fn compute_gradient_of_this_at_operand(
        &self,
        parameters: &[f64],
        operand_outputs: &[f64],
    ) -> Vec<f64> {
        assert!(parameters.is_empty());
        assert!(!operand_outputs.is_empty());
        let n = operand_outputs.len() / 2;
        let y = &operand_outputs[..n];
        let y_hat = &operand_outputs[n..];
        mse_derivative(y, y_hat)
    }

    fn compute_gradient_of_this_at_parameter(
        &self,
        parameters: &[f64],
        operand_outputs: &[f64],
    ) -> Vec<f64> {
        assert!(parameters.is_empty());
        assert!(!operand_outputs.is_empty());
        vec![]
    }
}

fn mse(y: &[f64], y_hat: &[f64]) -> f64 {
    assert_eq!(y.len(), y_hat.len());
    let n = y.len();
    let se = y
        .iter()
        .copied()
        .zip(y_hat.iter().copied())
        .map(|(y, y_hat)| (y - y_hat).powi(2));
    se.map(|x| x / n as f64).sum()
}

fn mse_derivative(y: &[f64], y_hat: &[f64]) -> Vec<f64> {
    assert_eq!(y.len(), y_hat.len());
    let n = y.len();
    let mut der = vec![];
    for (y, y_hat) in y.iter().copied().zip(y_hat.iter().copied()) {
        let d = 2.0 * (y - y_hat) / n as f64;
        der.push(d);
    }
    for (y, y_hat) in y.iter().copied().zip(y_hat.iter().copied()) {
        let d = -2.0 * (y - y_hat) / n as f64;
        der.push(d);
    }
    der
}
