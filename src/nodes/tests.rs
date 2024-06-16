use std::sync::Arc;

use crate::{
    computation::ComputationMode, mut_cell::MutCell, node::NodeContext,
    nodes::input::InputNodeBatchParams,
};

use super::{bias::bias_node, input::input_node_batch, relu::relu_node, weights::weight_node};

#[test]
fn linear_evaluate() {
    let input_nodes = input_node_batch(InputNodeBatchParams { start: 0, len: 3 });
    let inputs = vec![1.0, 2.0, 3.0];
    let initial_weights = vec![3.0, 2.0, 1.0];
    let initial_weights = Arc::new(MutCell::new(initial_weights));
    let initial_bias = 4.0;
    let initial_bias = Arc::new(MutCell::new(vec![initial_bias]));
    let weight_node = weight_node(input_nodes, initial_weights, None).unwrap();
    let mut bias_node = bias_node(Arc::new(MutCell::new(weight_node)), initial_bias);
    let mut cx = NodeContext::new();
    bias_node.evaluate_once(&[&inputs], &mut cx, ComputationMode::Inference);
    let output = bias_node.output().unwrap()[0];
    assert_eq!(output, (3.0 * 1.0 + 2.0 * 2.0 + 1.0 * 3.0) + 4.0);
}

#[test]
fn linear_gradient_of_this_at_operand() {
    let input_nodes = input_node_batch(InputNodeBatchParams { start: 0, len: 3 });
    let inputs = vec![1.0, 2.0, 3.0];
    let initial_weights = vec![3.0, 2.0, 1.0];
    let initial_weights = Arc::new(MutCell::new(initial_weights));
    let initial_bias = 4.0;
    let initial_bias = Arc::new(MutCell::new(vec![initial_bias]));
    let weight_node = weight_node(input_nodes, initial_weights, None).unwrap();
    let mut bias_node = bias_node(Arc::new(MutCell::new(weight_node)), initial_bias);
    let mut cx = NodeContext::new();
    bias_node.evaluate_once(&[&inputs], &mut cx, ComputationMode::Inference);
    let batch_index = 0;
    let grad = bias_node
        .gradient_of_this_at_operand(batch_index, &bias_node.parameters().borrow(), &mut cx)
        .unwrap();
    assert_eq!(&grad, &[1.0]);
}

#[test]
fn linear_gradient_of_this_at_parameter() {
    let input_nodes = input_node_batch(InputNodeBatchParams { start: 0, len: 3 });
    let inputs = vec![1.0, 2.0, 3.0];
    let initial_weights = vec![3.0, 2.0, 1.0];
    let initial_weights = Arc::new(MutCell::new(initial_weights));
    let initial_bias = 4.0;
    let initial_bias = Arc::new(MutCell::new(vec![initial_bias]));
    let weight_node = weight_node(input_nodes, initial_weights, None).unwrap();
    let mut bias_node = bias_node(Arc::new(MutCell::new(weight_node)), initial_bias);
    let mut cx = NodeContext::new();
    bias_node.evaluate_once(&[&inputs], &mut cx, ComputationMode::Inference);
    let batch_index = 0;
    let grad = bias_node
        .gradient_of_this_at_parameter(batch_index, &bias_node.parameters().borrow(), &mut cx)
        .unwrap();
    assert_eq!(&grad, &[1.0]);
}

#[test]
fn linear_with_relu_evaluate() {
    let input_nodes = input_node_batch(InputNodeBatchParams { start: 0, len: 3 });
    let inputs = vec![1.0, 2.0, 3.0];
    let initial_weights = vec![3.0, 2.0, 1.0];
    let initial_weights = Arc::new(MutCell::new(initial_weights));
    let initial_bias = -20.0;
    let initial_bias = Arc::new(MutCell::new(vec![initial_bias]));
    let weight_node = weight_node(input_nodes, initial_weights, None).unwrap();
    let bias_node = bias_node(Arc::new(MutCell::new(weight_node)), initial_bias);
    let bias_node = Arc::new(MutCell::new(bias_node));
    let mut relu_node = relu_node(Arc::clone(&bias_node));
    let mut cx = NodeContext::new();
    relu_node.evaluate_once(&[&inputs], &mut cx, ComputationMode::Inference);
    let output = relu_node.output().unwrap()[0];
    assert_eq!(output, 0.0);
    {
        let mut bias_node = bias_node.borrow_mut();
        bias_node.evaluate_once(&[&inputs], &mut cx, ComputationMode::Inference);
        let output = bias_node.output().unwrap()[0];
        assert_eq!(output, -10.0);
    }
}
