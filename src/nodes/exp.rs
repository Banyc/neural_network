use graph::NodeIdx;

use crate::{
    computation::{NodeBackpropagationComputation, NodeComputation, NodeScalarComputation},
    node::CompNode,
    param::empty_shared_params,
};

/// ```math
/// f(x) = a^x
/// ```
pub fn exp_node(operand: NodeIdx, base: f64) -> CompNode {
    let computation = ExpNodeComputation { base };
    CompNode::new(
        vec![operand],
        NodeComputation::Scalar(Box::new(computation)),
        empty_shared_params(),
    )
}

#[derive(Debug, Clone)]
struct ExpNodeComputation {
    base: f64,
}
impl NodeScalarComputation for ExpNodeComputation {
    fn compute_output(
        &self,
        parameters: &[f64],
        operand_outputs: &[f64],
        _graph_inputs: &[f64],
    ) -> f64 {
        assert!(parameters.is_empty());
        assert_eq!(operand_outputs.len(), 1);
        exp(self.base, operand_outputs[0])
    }
}
impl NodeBackpropagationComputation for ExpNodeComputation {
    fn compute_gradient_of_this_at_operand(
        &self,
        parameters: &[f64],
        operand_outputs: &[f64],
        mut buf: Vec<f64>,
    ) -> Vec<f64> {
        assert!(parameters.is_empty());
        assert_eq!(operand_outputs.len(), 1);
        buf.extend([exp_derivative(self.base, operand_outputs[0])]);
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

fn exp(base: f64, power: f64) -> f64 {
    base.powf(power)
}

fn exp_derivative(base: f64, power: f64) -> f64 {
    exp(base, power) * base.ln()
}
