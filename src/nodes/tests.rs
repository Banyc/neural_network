use std::sync::{Arc, Mutex};

use super::{
    bias_node::bias_node, input_node::input_node_batch, relu_node::relu_node,
    weight_node::weight_node,
};

#[test]
fn linear_evaluate() {
    let input_nodes = input_node_batch(3);
    let inputs = vec![1.0, 2.0, 3.0];
    let initial_weights = vec![3.0, 2.0, 1.0];
    let initial_bias = 4.0;
    let weight_node = weight_node(input_nodes, Some(initial_weights)).unwrap();
    let mut bias_node = bias_node(Arc::new(Mutex::new(weight_node)), Some(initial_bias));
    let ret = bias_node.evaluate(&inputs);
    assert_eq!(ret, (3.0 * 1.0 + 2.0 * 2.0 + 1.0 * 3.0) + 4.0);
}

#[test]
fn linear_local_operand_gradient() {
    let input_nodes = input_node_batch(3);
    let inputs = vec![1.0, 2.0, 3.0];
    let initial_weights = vec![3.0, 2.0, 1.0];
    let initial_bias = 4.0;
    let weight_node = weight_node(input_nodes, Some(initial_weights)).unwrap();
    let mut bias_node = bias_node(Arc::new(Mutex::new(weight_node)), Some(initial_bias));
    bias_node.evaluate(&inputs);
    let ret = bias_node.local_operand_gradient().unwrap();
    assert_eq!(ret.as_ref(), &vec![1.0]);
}

#[test]
fn linear_local_parameter_gradient() {
    let input_nodes = input_node_batch(3);
    let inputs = vec![1.0, 2.0, 3.0];
    let initial_weights = vec![3.0, 2.0, 1.0];
    let initial_bias = 4.0;
    let weight_node = weight_node(input_nodes, Some(initial_weights)).unwrap();
    let mut bias_node = bias_node(Arc::new(Mutex::new(weight_node)), Some(initial_bias));
    bias_node.evaluate(&inputs);
    let ret = bias_node.local_parameter_gradient().unwrap();
    assert_eq!(ret.as_ref(), &vec![1.0]);
}

#[test]
fn linear_with_relu_evaluate() {
    let input_nodes = input_node_batch(3);
    let inputs = vec![1.0, 2.0, 3.0];
    let initial_weights = vec![3.0, 2.0, 1.0];
    let initial_bias = -20.0;
    let weight_node = weight_node(input_nodes, Some(initial_weights)).unwrap();
    let bias_node = bias_node(Arc::new(Mutex::new(weight_node)), Some(initial_bias));
    let bias_node = Arc::new(Mutex::new(bias_node));
    let mut relu_node = relu_node(Arc::clone(&bias_node));
    let ret = relu_node.evaluate(&inputs);
    assert_eq!(ret, 0.0);
    {
        let mut bias_node = bias_node.lock().unwrap();
        let ret = bias_node.evaluate(&inputs);
        assert_eq!(ret, -10.0);
    }
}
