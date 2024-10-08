use graph::NodeIdx;
use primitive::{vec_seg::SegKey, Len};

use crate::{
    computation::{NodeBackpropagationComputation, NodeComputation, NodeScalarComputation},
    node::CompNode,
};

pub fn default_bias() -> f64 {
    0.0
}

/// ```math
/// f_b (x) = x + b
/// ```
pub fn bias_node(operand: NodeIdx, bias: SegKey) -> CompNode {
    let computation = BiasNodeComputation {};
    assert_eq!(bias.len(), 1);
    CompNode::new(
        vec![operand],
        NodeComputation::Scalar(Box::new(computation)),
        bias,
    )
}

#[derive(Debug, Clone)]
struct BiasNodeComputation {}
impl NodeScalarComputation for BiasNodeComputation {
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
}
impl NodeBackpropagationComputation for BiasNodeComputation {
    fn compute_gradient_of_this_at_operand(
        &self,
        parameters: &[f64],
        operand_outputs: &[f64],
        mut buf: Vec<f64>,
    ) -> Vec<f64> {
        assert_eq!(operand_outputs.len(), 1);
        assert_eq!(parameters.len(), 1);
        buf.extend([bias_derivative()]);
        buf
    }

    fn compute_gradient_of_this_at_parameter(
        &self,
        parameters: &[f64],
        operand_outputs: &[f64],
        mut buf: Vec<f64>,
    ) -> Vec<f64> {
        assert_eq!(operand_outputs.len(), 1);
        assert_eq!(parameters.len(), 1);
        buf.extend([1.0]);
        buf
    }
}

fn bias(x: f64, b: f64) -> f64 {
    x + b
}

fn bias_derivative() -> f64 {
    1.0
}
