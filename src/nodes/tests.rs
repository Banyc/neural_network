use std::sync::Arc;

use parking_lot::Mutex;

use crate::nodes::input::InputNodeBatchParams;

use super::{bias::bias_node, input::input_node_batch, relu::relu_node, weights::weight_node};

#[test]
fn linear_evaluate() {
    let input_nodes = input_node_batch(InputNodeBatchParams { start: 0, len: 3 });
    let inputs = vec![1.0, 2.0, 3.0];
    let initial_weights = vec![3.0, 2.0, 1.0];
    let initial_weights = Arc::new(Mutex::new(initial_weights));
    let initial_bias = 4.0;
    let initial_bias = Arc::new(Mutex::new(vec![initial_bias]));
    let weight_node = weight_node(input_nodes, Some(initial_weights), None).unwrap();
    let mut bias_node = bias_node(Arc::new(Mutex::new(weight_node)), Some(initial_bias));
    let batch_index = 0;
    let ret = bias_node.evaluate_once(&inputs, batch_index);
    assert_eq!(ret, (3.0 * 1.0 + 2.0 * 2.0 + 1.0 * 3.0) + 4.0);
}

#[test]
fn linear_gradient_of_this_at_operand() {
    let input_nodes = input_node_batch(InputNodeBatchParams { start: 0, len: 3 });
    let inputs = vec![1.0, 2.0, 3.0];
    let initial_weights = vec![3.0, 2.0, 1.0];
    let initial_weights = Arc::new(Mutex::new(initial_weights));
    let initial_bias = 4.0;
    let initial_bias = Arc::new(Mutex::new(vec![initial_bias]));
    let weight_node = weight_node(input_nodes, Some(initial_weights), None).unwrap();
    let mut bias_node = bias_node(Arc::new(Mutex::new(weight_node)), Some(initial_bias));
    let batch_index = 0;
    bias_node.evaluate_once(&inputs, batch_index);
    let ret = bias_node
        .gradient_of_this_at_operand(batch_index, &bias_node.parameters().lock(), vec![])
        .unwrap();
    assert_eq!(&ret, &[1.0]);
}

#[test]
fn linear_gradient_of_this_at_parameter() {
    let input_nodes = input_node_batch(InputNodeBatchParams { start: 0, len: 3 });
    let inputs = vec![1.0, 2.0, 3.0];
    let initial_weights = vec![3.0, 2.0, 1.0];
    let initial_weights = Arc::new(Mutex::new(initial_weights));
    let initial_bias = 4.0;
    let initial_bias = Arc::new(Mutex::new(vec![initial_bias]));
    let weight_node = weight_node(input_nodes, Some(initial_weights), None).unwrap();
    let mut bias_node = bias_node(Arc::new(Mutex::new(weight_node)), Some(initial_bias));
    let batch_index = 0;
    bias_node.evaluate_once(&inputs, batch_index);
    let ret = bias_node
        .gradient_of_this_at_parameter(batch_index, &bias_node.parameters().lock(), vec![])
        .unwrap();
    assert_eq!(&ret, &[1.0]);
}

#[test]
fn linear_with_relu_evaluate() {
    let input_nodes = input_node_batch(InputNodeBatchParams { start: 0, len: 3 });
    let inputs = vec![1.0, 2.0, 3.0];
    let initial_weights = vec![3.0, 2.0, 1.0];
    let initial_weights = Arc::new(Mutex::new(initial_weights));
    let initial_bias = -20.0;
    let initial_bias = Arc::new(Mutex::new(vec![initial_bias]));
    let weight_node = weight_node(input_nodes, Some(initial_weights), None).unwrap();
    let bias_node = bias_node(Arc::new(Mutex::new(weight_node)), Some(initial_bias));
    let bias_node = Arc::new(Mutex::new(bias_node));
    let mut relu_node = relu_node(Arc::clone(&bias_node));
    let batch_index = 0;
    let ret = relu_node.evaluate_once(&inputs, batch_index);
    assert_eq!(ret, 0.0);
    {
        let mut bias_node = bias_node.lock();
        let ret = bias_node.evaluate_once(&inputs, batch_index);
        assert_eq!(ret, -10.0);
    }
}
