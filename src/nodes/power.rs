use graph::NodeIdx;

use crate::{
    computation::{NodeBackpropagationComputation, NodeComputation, NodeScalarComputation},
    node::CompNode,
    param::empty_shared_params,
};

/// ```math
/// f(x) = x^a
/// ```
pub fn power_node(operand: NodeIdx, power: f64) -> CompNode {
    let computation = PowerNodeComputation { power };
    CompNode::new(
        vec![operand],
        NodeComputation::Scalar(Box::new(computation)),
        empty_shared_params(),
    )
}

#[derive(Debug, Clone)]
struct PowerNodeComputation {
    power: f64,
}
impl NodeScalarComputation for PowerNodeComputation {
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
}
impl NodeBackpropagationComputation for PowerNodeComputation {
    fn compute_gradient_of_this_at_operand(
        &self,
        parameters: &[f64],
        operand_outputs: &[f64],
        mut buf: Vec<f64>,
    ) -> Vec<f64> {
        assert!(parameters.is_empty());
        assert_eq!(operand_outputs.len(), 1);
        buf.extend([power_derivative(operand_outputs[0], self.power)]);
        buf
    }

    fn compute_gradient_of_this_at_parameter(
        &self,
        parameters: &[f64],
        operand_outputs: &[f64],
        buf: Vec<f64>,
    ) -> Vec<f64> {
        assert!(parameters.is_empty());
        assert_eq!(operand_outputs.len(), 1);
        buf
    }
}

fn power(base: f64, power: f64) -> f64 {
    base.powf(power)
}

fn power_derivative(base: f64, power: f64) -> f64 {
    power * base.powf(power - 1.)
}
