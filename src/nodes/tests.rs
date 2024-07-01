use graph::dependency_order;

use crate::{
    computation::ComputationMode,
    mut_cell::MutCell,
    node::{evaluate_once, CompNode, GraphBuilder, NodeContext},
    nodes::input::InputNodeBatchParams,
    ref_ctr::RefCtr,
};

use super::{bias::bias_node, input::input_node_batch, relu::relu_node, weights::weight_node};

fn assertion(assert_node: impl Fn(&CompNode, &mut NodeContext)) {
    let mut graph = GraphBuilder::new();
    let input_nodes =
        graph.insert_nodes(input_node_batch(InputNodeBatchParams { start: 0, len: 3 }));
    let inputs = vec![1.0, 2.0, 3.0];
    let initial_weights = vec![3.0, 2.0, 1.0];
    let initial_weights = RefCtr::new(MutCell::new(initial_weights));
    let initial_bias = 4.0;
    let initial_bias = RefCtr::new(MutCell::new(vec![initial_bias]));
    let weight_node = graph.insert_node(weight_node(input_nodes, initial_weights, None).unwrap());
    let bias_node = graph.insert_node(bias_node(weight_node, initial_bias));
    let mut graph = graph.build();
    let nodes_forward = dependency_order(&graph, &[bias_node]);
    let mut cx = NodeContext::new();
    evaluate_once(
        &mut graph,
        &nodes_forward,
        &[inputs],
        &mut cx,
        ComputationMode::Inference,
    );
    let bias_node = graph.nodes().get(bias_node).unwrap();
    assert_node(bias_node, &mut cx);
}

#[test]
fn linear_evaluate() {
    assertion(|node, _| {
        let output = node.output().unwrap()[0];
        assert_eq!(output, (3.0 * 1.0 + 2.0 * 2.0 + 1.0 * 3.0) + 4.0);
    });
}

#[test]
fn linear_gradient_of_this_at_operand() {
    assertion(|node, cx| {
        let batch_index = 0;
        let grad = node
            .gradient_of_this_at_operand(batch_index, &node.parameters().borrow(), cx)
            .unwrap();
        assert_eq!(&grad, &[1.0]);
    });
}

#[test]
fn linear_gradient_of_this_at_parameter() {
    assertion(|node, cx| {
        let batch_index = 0;
        let grad = node
            .gradient_of_this_at_parameter(batch_index, &node.parameters().borrow(), cx)
            .unwrap();
        assert_eq!(&grad, &[1.0]);
    });
}

#[test]
fn linear_with_relu_evaluate() {
    let mut graph = GraphBuilder::new();
    let input_nodes =
        graph.insert_nodes(input_node_batch(InputNodeBatchParams { start: 0, len: 3 }));
    let inputs = vec![1.0, 2.0, 3.0];
    let initial_weights = vec![3.0, 2.0, 1.0];
    let initial_weights = RefCtr::new(MutCell::new(initial_weights));
    let initial_bias = -20.0;
    let initial_bias = RefCtr::new(MutCell::new(vec![initial_bias]));
    let weight_node = graph.insert_node(weight_node(input_nodes, initial_weights, None).unwrap());
    let bias_node = graph.insert_node(bias_node(weight_node, initial_bias));
    let relu_node = graph.insert_node(relu_node(bias_node));
    let mut graph = graph.build();
    let mut cx = NodeContext::new();

    let nodes_forward = dependency_order(&graph, &[relu_node]);
    evaluate_once(
        &mut graph,
        &nodes_forward,
        &[inputs],
        &mut cx,
        ComputationMode::Inference,
    );
    {
        let output = graph.nodes().get(relu_node).unwrap().output().unwrap()[0];
        assert_eq!(output, 0.0);
    }
    {
        let output = graph.nodes().get(bias_node).unwrap().output().unwrap()[0];
        assert_eq!(output, -10.0);
    }
}
