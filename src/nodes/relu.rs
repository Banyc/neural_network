use graph::NodeIdx;

use crate::{
    computation::{NodeBackpropagationComputation, NodeComputation, NodeScalarComputation},
    mut_cell::MutCell,
    node::CompNode,
    param::empty_shared_params,
    ref_ctr::RefCtr,
};

/// ```math
/// f(x) = \begin{cases}
///   x & x \geq 0 \\
///   0 & x < 0 \\
/// \end{cases}
/// ```
pub fn relu_node(operand: NodeIdx) -> CompNode {
    let computation = ReluNodeComputation {};
    CompNode::new(
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
    use graph::dependency_order;

    use crate::{
        computation::ComputationMode,
        node::{evaluate_once, GraphBuilder, NodeContext},
        nodes::input::input_node,
        param::Params,
    };

    use super::*;

    fn assertion(input: f64, assert_relu: impl Fn(&CompNode, &Params, &mut NodeContext)) {
        let mut params = Params::new();
        let mut graph = GraphBuilder::new();
        let input_node = graph.insert_node(input_node(0));
        let relu = graph.insert_node(relu_node(input_node));
        let mut graph = graph.build();
        let nodes_forward = dependency_order(&graph, &[relu]);
        let mut cx = NodeContext::new();
        evaluate_once(
            &mut graph,
            &nodes_forward,
            &mut params,
            &[&[input]],
            &mut cx,
            ComputationMode::Inference,
        );
        let relu = graph.nodes().get(relu).unwrap();
        assert_relu(relu, &params, &mut cx);
    }

    #[test]
    fn evaluate_negative() {
        assertion(-2.0, |relu, _, _| {
            let output = relu.output().unwrap()[0];
            assert_eq!(output, 0.0);
        });
    }

    #[test]
    fn evaluate_positive() {
        assertion(3.0, |relu, _, _| {
            let output = relu.output().unwrap()[0];
            assert_eq!(output, 3.0);
        });
    }

    #[test]
    fn positive_gradient_of_this_at_operand() {
        assertion(3.0, |relu, params, cx| {
            let batch_index = 0;
            let ret = relu
                .gradient_of_this_at_operand(batch_index, params.seg().slice(relu.parameters()), cx)
                .unwrap();
            assert_eq!(ret[0], 1.0);
        });
    }

    #[test]
    fn negative_gradient_of_this_at_operand() {
        assertion(-3.0, |relu, params, cx| {
            let batch_index = 0;
            let ret = relu
                .gradient_of_this_at_operand(batch_index, params.seg().slice(relu.parameters()), cx)
                .unwrap();
            assert_eq!(ret[0], 0.0);
        });
    }

    #[test]
    fn empty_gradient_of_this_at_parameter() {
        assertion(3.0, |relu, params, cx| {
            let batch_index = 0;
            let ret = relu
                .gradient_of_this_at_parameter(
                    batch_index,
                    params.seg().slice(relu.parameters()),
                    cx,
                )
                .unwrap();
            assert_eq!(ret.len(), 0);
        });
    }
}
