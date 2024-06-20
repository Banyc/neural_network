use crate::{
    computation::{NodeBackpropagationComputation, NodeComputation, NodeScalarComputation},
    mut_cell::MutCell,
    node::{Node, SharedNode},
    param::empty_shared_params,
    ref_ctr::RefCtr,
};

/// ```math
/// f(\mu, x) = \sqrt{\frac{\sum (x - \mu)^2}{n}}
/// ```
pub fn std_dev_node(operands: Vec<SharedNode>) -> Node {
    assert!(2 <= operands.len());
    let computation = StdDevNodeComputation {};
    Node::new(
        operands,
        RefCtr::new(MutCell::new(NodeComputation::Scalar(Box::new(computation)))),
        empty_shared_params(),
    )
}

#[derive(Debug)]
struct StdDevNodeComputation {}
impl NodeScalarComputation for StdDevNodeComputation {
    fn compute_output(
        &self,
        parameters: &[f64],
        operand_outputs: &[f64],
        _graph_inputs: &[f64],
    ) -> f64 {
        assert!(parameters.is_empty());
        assert!(2 <= operand_outputs.len());
        let mean = operand_outputs[0];
        std_dev(mean, &operand_outputs[1..])
    }
}
impl NodeBackpropagationComputation for StdDevNodeComputation {
    fn compute_gradient_of_this_at_operand(
        &self,
        parameters: &[f64],
        operand_outputs: &[f64],
        buf: Vec<f64>,
    ) -> Vec<f64> {
        assert!(parameters.is_empty());
        assert!(2 <= operand_outputs.len());
        let mean = operand_outputs[0];
        std_dev_derivative(mean, &operand_outputs[1..], buf)
    }

    fn compute_gradient_of_this_at_parameter(
        &self,
        parameters: &[f64],
        operand_outputs: &[f64],
        buf: Vec<f64>,
    ) -> Vec<f64> {
        assert!(parameters.is_empty());
        assert!(2 <= operand_outputs.len());
        buf
    }
}

fn std_dev(mean: f64, x: &[f64]) -> f64 {
    let n = x.len();
    x.iter()
        .copied()
        .map(|x| (x - mean).powi(2) / n as f64)
        .map(|x| x.sqrt())
        .sum::<f64>()
}

fn std_dev_derivative(mean: f64, x: &[f64], mut buf: Vec<f64>) -> Vec<f64> {
    let std_dev = std_dev(mean, x);
    let n = x.len();
    let x_der = x.iter().copied().map(|x| (x - mean) / (n as f64 * std_dev));
    buf.push(0.);
    buf.extend(x_der);
    buf
}
