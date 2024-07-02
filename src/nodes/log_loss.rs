use graph::NodeIdx;

use crate::{
    computation::{NodeBackpropagationComputation, NodeComputation, NodeScalarComputation},
    node::CompNode,
    param::empty_shared_params,
};

/// ```math
/// f(y, \hat{y}) = -\frac{1}{n} \sum y \log(\hat{y}) + (1 - y) \log(1 - \hat{y})
/// ```
pub fn log_loss_node(operands: Vec<NodeIdx>) -> CompNode {
    assert!(!operands.is_empty());
    assert!(operands.len() % 2 == 0);
    let computation = MseNodeComputation {};
    CompNode::new(
        operands,
        NodeComputation::Scalar(Box::new(computation)),
        empty_shared_params(),
    )
}

#[derive(Debug, Clone)]
struct MseNodeComputation {}
impl NodeScalarComputation for MseNodeComputation {
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
        log_loss(y, y_hat)
    }
}
impl NodeBackpropagationComputation for MseNodeComputation {
    fn compute_gradient_of_this_at_operand(
        &self,
        parameters: &[f64],
        operand_outputs: &[f64],
        buf: Vec<f64>,
    ) -> Vec<f64> {
        assert!(parameters.is_empty());
        assert!(!operand_outputs.is_empty());
        let n = operand_outputs.len() / 2;
        let y = &operand_outputs[..n];
        let y_hat = &operand_outputs[n..];
        log_loss_derivative(y, y_hat, buf)
    }

    fn compute_gradient_of_this_at_parameter(
        &self,
        parameters: &[f64],
        operand_outputs: &[f64],
        buf: Vec<f64>,
    ) -> Vec<f64> {
        assert!(parameters.is_empty());
        assert!(!operand_outputs.is_empty());
        buf
    }
}

fn log_loss(y: &[f64], y_hat: &[f64]) -> f64 {
    assert_eq!(y.len(), y_hat.len());
    for y_hat in y_hat {
        assert!((0. ..=1.).contains(y_hat));
    }
    for &y in y {
        assert!(y == 0. || y == 1.);
    }
    let n = y.len();
    let se = y
        .iter()
        .copied()
        .zip(y_hat.iter().copied())
        .map(|(y, y_hat)| y * y_hat.ln() + (1. - y) * (1. - y_hat).ln());
    se.map(|x| -x / n as f64).sum()
}

fn log_loss_derivative(y: &[f64], y_hat: &[f64], mut buf: Vec<f64>) -> Vec<f64> {
    assert_eq!(y.len(), y_hat.len());
    let n = y.len();
    for (_y, y_hat) in y.iter().copied().zip(y_hat.iter().copied()) {
        let d = -(1. / n as f64) * (y_hat.ln() - (1. - y_hat).ln());
        buf.push(d);
    }
    for (y, y_hat) in y.iter().copied().zip(y_hat.iter().copied()) {
        let d = -(1. / n as f64) * ((y / y_hat) - (1. - y) / (1. - y_hat));
        buf.push(d);
    }
    buf
}
