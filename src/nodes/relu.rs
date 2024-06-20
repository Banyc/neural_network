use crate::{
    computation::{NodeBackpropagationComputation, NodeComputation, NodeScalarComputation},
    mut_cell::MutCell,
    node::{Node, SharedNode},
    param::empty_shared_params,
    ref_ctr::RefCtr,
};

/// ```math
/// f(x) = \begin{cases}
///   x & x \geq 0 \\
///   0 & x < 0 \\
/// \end{cases}
/// ```
pub fn relu_node(operand: SharedNode) -> Node {
    let computation = ReluNodeComputation {};
    Node::new(
        vec![operand],
        RefCtr::new(MutCell::new(NodeComputation::Scalar(Box::new(computation)))),
        empty_shared_params(),
    )
}

#[derive(Debug)]
struct ReluNodeComputation {}
impl NodeScalarComputation for ReluNodeComputation {
    fn compute_output(
        &self,
        parameters: &[f64],
        operand_outputs: &[f64],
        _graph_inputs: &[f64],
    ) -> f64 {
        assert!(parameters.is_empty());
        assert_eq!(operand_outputs.len(), 1);
        relu(operand_outputs[0])
    }
}
impl NodeBackpropagationComputation for ReluNodeComputation {
    fn compute_gradient_of_this_at_operand(
        &self,
        parameters: &[f64],
        operand_outputs: &[f64],
        mut buf: Vec<f64>,
    ) -> Vec<f64> {
        assert!(parameters.is_empty());
        assert_eq!(operand_outputs.len(), 1);
        buf.extend([relu_derivative(operand_outputs[0])]);
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

fn relu(x: f64) -> f64 {
    f64::max(x, 0.0)
}

fn relu_derivative(x: f64) -> f64 {
    match x {
        _ if x > 0.0 => 1.0,
        _ => 0.0,
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        computation::ComputationMode, mut_cell::MutCell, node::NodeContext,
        nodes::input::input_node,
    };

    use super::*;

    #[test]
    fn evaluate_negative() {
        let input_node = input_node(0);
        let mut relu = relu_node(RefCtr::new(MutCell::new(input_node)));
        let mut cx = NodeContext::new();
        relu.evaluate_once(&[&[-2.0]], &mut cx, ComputationMode::Inference);
        let output = relu.output().unwrap()[0];
        assert!(output >= 0.0);
        assert!(output <= 0.0);
    }

    #[test]
    fn evaluate_positive() {
        let input_node = input_node(0);
        let mut relu = relu_node(RefCtr::new(MutCell::new(input_node)));
        let mut cx = NodeContext::new();
        relu.evaluate_once(&[&[3.0]], &mut cx, ComputationMode::Inference);
        let output = relu.output().unwrap()[0];
        assert!(output >= 3.0);
        assert!(output <= 3.0);
    }

    #[test]
    fn positive_gradient_of_this_at_operand() {
        let input_node = input_node(0);
        let mut relu = relu_node(RefCtr::new(MutCell::new(input_node)));
        let mut cx = NodeContext::new();
        relu.evaluate_once(&[&[3.0]], &mut cx, ComputationMode::Inference);
        let batch_index = 0;
        let ret = relu
            .gradient_of_this_at_operand(batch_index, &relu.parameters().borrow(), &mut cx)
            .unwrap();
        assert!(ret[0] >= 1.0);
        assert!(ret[0] <= 1.0);
    }

    #[test]
    fn negative_gradient_of_this_at_operand() {
        let input_node = input_node(0);
        let mut relu = relu_node(RefCtr::new(MutCell::new(input_node)));
        let mut cx = NodeContext::new();
        relu.evaluate_once(&[&[-3.0]], &mut cx, ComputationMode::Inference);
        let batch_index = 0;
        let ret = relu
            .gradient_of_this_at_operand(batch_index, &relu.parameters().borrow(), &mut cx)
            .unwrap();
        assert!(ret[0] >= 0.0);
        assert!(ret[0] <= 0.0);
    }

    #[test]
    fn empty_gradient_of_this_at_parameter() {
        let input_node = input_node(0);
        let mut relu = relu_node(RefCtr::new(MutCell::new(input_node)));
        let mut cx = NodeContext::new();
        relu.evaluate_once(&[&[3.0]], &mut cx, ComputationMode::Inference);
        let batch_index = 0;
        let ret = relu
            .gradient_of_this_at_parameter(batch_index, &relu.parameters().borrow(), &mut cx)
            .unwrap();
        assert_eq!(ret.len(), 0);
    }
}
