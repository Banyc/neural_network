use std::sync::{Arc, Mutex};

use crate::{
    neural_network::{EvalOption, NeuralNetwork, TrainOption},
    node::{clone_node_batch, Node},
    nodes::{
        bias_node::bias_node,
        input_node::{input_node, input_node_batch},
        l2_error_node::l2_error_node,
        linear_node::linear_node,
        relu_node::relu_node,
        sigmoid_node::sigmoid_node,
        weight_node::weight_node,
    },
};

fn single_linear_relu(
    input_nodes: Vec<Arc<Mutex<Node>>>,
    initial_weights: Option<Vec<f64>>,
    initial_bias: Option<f64>,
) -> Node {
    let linear_node = linear_node(input_nodes, initial_weights, initial_bias, None, None).unwrap();
    relu_node(linear_node)
}

fn single_linear_relu_network(
    node_count: usize,
    initial_weights: Option<Vec<f64>>,
    initial_bias: Option<f64>,
) -> NeuralNetwork {
    let input_nodes = input_node_batch(node_count);
    let relu_node = single_linear_relu(input_nodes, initial_weights, initial_bias);
    let relu_node = Arc::new(Mutex::new(relu_node));
    let label_node = input_node(node_count);
    let error_node = l2_error_node(Arc::clone(&relu_node), Arc::new(Mutex::new(label_node)));
    NeuralNetwork::new(
        vec![relu_node],
        Arc::new(Mutex::new(error_node)),
        vec![node_count],
        1e-2,
    )
}

#[test]
fn evaluate() {
    let mut network = single_linear_relu_network(3, Some(vec![3.0, 2.0, 1.0]), Some(-20.0));
    let ret = network.evaluate(&[1.0, 2.0, 3.0]);
    assert_eq!(ret[0], 0.0);
}

#[test]
fn error() {
    let input = input_node(0);
    let relu = relu_node(Arc::new(Mutex::new(input)));
    let relu = Arc::new(Mutex::new(relu));
    let label = input_node(1);
    let error = l2_error_node(Arc::clone(&relu), Arc::new(Mutex::new(label)));
    let error = Arc::new(Mutex::new(error));
    let mut network = NeuralNetwork::new(vec![relu], error, vec![1], 1e-2);
    let inputs = vec![-2.0, 1.0];
    let ret = network.evaluate(&inputs);
    assert_eq!(ret[0], 0.0);
    let batch_index = 0;
    let ret = network.compute_error(&inputs, EvalOption::ClearCache, batch_index);
    assert_eq!(ret, 1.0);
}

#[test]
fn cache_reset() {
    let mut network = single_linear_relu_network(2, Some(vec![2.0, 1.0]), Some(3.0));
    let ret = network.evaluate(&[2.0, -2.0]);
    assert_eq!(ret[0], 5.0);
    let ret = network.evaluate(&[6.0, -2.0]);
    assert!(ret[0] != 5.0);
}

#[test]
fn errors_on_dataset() {
    let mut network = single_linear_relu_network(2, Some(vec![2.0, 1.0]), Some(3.0));
    let dataset = vec![vec![2.0, -2.0, 5.0], vec![6.0, -2.0, 5.0]];
    let ret = network.accuracy(&dataset, binary_accurate);
    assert!(ret > 0.499);
    assert!(ret < 0.501);
}

#[test]
fn gradients() {
    let label_index = 2;
    let input_nodes = input_node_batch(label_index);
    let initial_weights = vec![2.0, 1.0];
    let weight_node = weight_node(input_nodes, Some(initial_weights), None).unwrap();
    let weight_node = Arc::new(Mutex::new(weight_node));
    let initial_bias = 3.0;
    let bias_node = bias_node(Arc::clone(&weight_node), Some(initial_bias));
    let bias_node = Arc::new(Mutex::new(bias_node));
    let relu_node = relu_node(Arc::clone(&bias_node));
    let relu_node = Arc::new(Mutex::new(relu_node));
    let label_node = input_node(label_index);
    let label_node = Arc::new(Mutex::new(label_node));
    let error_node = l2_error_node(Arc::clone(&relu_node), label_node);
    let error_node = Arc::new(Mutex::new(error_node));
    let mut network = NeuralNetwork::new(
        vec![Arc::clone(&relu_node)],
        Arc::clone(&error_node),
        vec![label_index],
        1e-2,
    );

    let inputs = vec![2.0, -2.0, 1.0];

    let ret = network.evaluate(&inputs);
    assert_eq!(ret[0], 5.0);
    let batch_index = 0;
    let ret = network.compute_error(&inputs, EvalOption::KeepCache, batch_index);
    assert_eq!(ret, 16.0);
}

#[test]
fn backpropagation_step() {
    let label_index = 2;
    let input_nodes = input_node_batch(label_index);
    let initial_weights = vec![2.0, 1.0];
    let weight_node = weight_node(input_nodes, Some(initial_weights), None).unwrap();
    let weight_node = Arc::new(Mutex::new(weight_node));
    let initial_bias = 3.0;
    let bias_node = bias_node(Arc::clone(&weight_node), Some(initial_bias));
    let bias_node = Arc::new(Mutex::new(bias_node));
    let relu_node = relu_node(Arc::clone(&bias_node));
    let relu_node = Arc::new(Mutex::new(relu_node));
    let label_node = input_node(label_index);
    let label_node = Arc::new(Mutex::new(label_node));
    let error_node = l2_error_node(Arc::clone(&relu_node), label_node);
    let error_node = Arc::new(Mutex::new(error_node));
    let step_size = 0.5;
    let mut network = NeuralNetwork::new(
        vec![Arc::clone(&relu_node)],
        Arc::clone(&error_node),
        vec![label_index],
        step_size,
    );

    let inputs = vec![2.0, -2.0, 1.0];
    network.backpropagation_step(&[&inputs]);
    {
        let weight_node = weight_node.lock().unwrap();
        assert_eq!(weight_node.parameters(), &[-6.0, 9.0])
    }
    {
        let bias_node = bias_node.lock().unwrap();
        assert_eq!(bias_node.parameters(), &[-1.0])
    }
}

#[test]
fn backpropagation_step2() {
    let label_index = 1;
    let input_nodes = input_node_batch(label_index);
    let initial_weights1 = vec![2.0];
    let weight_node1 = weight_node(input_nodes, Some(initial_weights1), None).unwrap();
    let weight_node1 = Arc::new(Mutex::new(weight_node1));
    let initial_weights2 = vec![3.0];
    let weight_node2 = weight_node(
        vec![Arc::clone(&weight_node1)],
        Some(initial_weights2),
        None,
    )
    .unwrap();
    let weight_node2 = Arc::new(Mutex::new(weight_node2));
    let label_node = input_node(label_index);
    let label_node = Arc::new(Mutex::new(label_node));
    let error_node = l2_error_node(Arc::clone(&weight_node2), label_node);
    let error_node = Arc::new(Mutex::new(error_node));
    let step_size = 0.5;
    let mut network = NeuralNetwork::new(
        vec![Arc::clone(&weight_node2)],
        Arc::clone(&error_node),
        vec![label_index],
        step_size,
    );

    let inputs = vec![2.0, 1.0];
    network.backpropagation_step(&[&inputs]);
    {
        let weight_node = weight_node2.lock().unwrap();
        assert_eq!(weight_node.parameters(), &[-41.0]); // 3 - 0.5 * 88
    }
    {
        let weight_node = weight_node1.lock().unwrap();
        assert_eq!(weight_node.parameters(), &[-64.0]); // 2 - 0.5 * 121
    }
}

#[test]
fn learn_xor_sigmoid() {
    let label_index = 2;
    let input_nodes = input_node_batch(label_index);
    let linear_node_1 =
        linear_node(clone_node_batch(&input_nodes), None, None, None, None).unwrap();
    let linear_node_2 =
        linear_node(clone_node_batch(&input_nodes), None, None, None, None).unwrap();
    let linear_node_3 =
        linear_node(clone_node_batch(&input_nodes), None, None, None, None).unwrap();
    let sigmoid_node_1 = sigmoid_node(linear_node_1);
    let sigmoid_node_2 = sigmoid_node(linear_node_2);
    let sigmoid_node_3 = sigmoid_node(linear_node_3);
    let sigmoid_nodes = vec![
        Arc::new(Mutex::new(sigmoid_node_1)),
        Arc::new(Mutex::new(sigmoid_node_2)),
        Arc::new(Mutex::new(sigmoid_node_3)),
    ];
    let linear_output = linear_node(sigmoid_nodes, None, None, None, None).unwrap();
    let output = sigmoid_node(linear_output);
    let output = Arc::new(Mutex::new(output));
    let label_node = input_node(label_index);
    let error_node = l2_error_node(Arc::clone(&output), Arc::new(Mutex::new(label_node)));
    let step_size = 0.5;
    let mut network = NeuralNetwork::new(
        vec![output],
        Arc::new(Mutex::new(error_node)),
        vec![label_index],
        step_size,
    );

    let dataset = vec![
        vec![0.0, 0.0, 0.0],
        vec![0.0, 1.0, 1.0],
        vec![1.0, 0.0, 1.0],
        vec![1.0, 1.0, 0.0],
    ];

    let max_steps = 10_000;
    network.train(&dataset, max_steps, TrainOption::StochasticGradientDescent);
    for inputs in &dataset {
        let ret = network.evaluate(inputs);
        assert!((ret[0] - inputs[label_index]).abs() < 0.1);
    }
    let ret = network.accuracy(&dataset, binary_accurate);
    assert_eq!(ret, 1.0);
}

#[test]
fn learn_xor_regularized_sigmoid() {
    let label_index = 2;
    let lambda = 0.0001;
    let input_nodes = input_node_batch(label_index);
    let linear_node_1 = linear_node(
        clone_node_batch(&input_nodes),
        None,
        None,
        Some(lambda),
        None,
    )
    .unwrap();
    let linear_node_2 = linear_node(
        clone_node_batch(&input_nodes),
        None,
        None,
        Some(lambda),
        None,
    )
    .unwrap();
    let linear_node_3 = linear_node(
        clone_node_batch(&input_nodes),
        None,
        None,
        Some(lambda),
        None,
    )
    .unwrap();
    let sigmoid_node_1 = sigmoid_node(linear_node_1);
    let sigmoid_node_2 = sigmoid_node(linear_node_2);
    let sigmoid_node_3 = sigmoid_node(linear_node_3);
    let sigmoid_nodes = vec![
        Arc::new(Mutex::new(sigmoid_node_1)),
        Arc::new(Mutex::new(sigmoid_node_2)),
        Arc::new(Mutex::new(sigmoid_node_3)),
    ];
    let linear_output = linear_node(sigmoid_nodes, None, None, Some(lambda), None).unwrap();
    let output = sigmoid_node(linear_output);
    let output = Arc::new(Mutex::new(output));
    let label_node = input_node(label_index);
    let error_node = l2_error_node(Arc::clone(&output), Arc::new(Mutex::new(label_node)));
    let step_size = 0.5;
    let mut network = NeuralNetwork::new(
        vec![output],
        Arc::new(Mutex::new(error_node)),
        vec![label_index],
        step_size,
    );

    let dataset = vec![
        vec![-1.0, -1.0, 0.0],
        vec![-1.0, 1.0, 1.0],
        vec![1.0, -1.0, 1.0],
        vec![1.0, 1.0, 0.0],
    ];

    let max_steps = 10_000;
    network.train(&dataset, max_steps, TrainOption::StochasticGradientDescent);
    for inputs in &dataset {
        let ret = network.evaluate(inputs);
        assert!((ret[0] - inputs[label_index]).abs() < 0.1);
    }
    let ret = network.accuracy(&dataset, binary_accurate);
    assert_eq!(ret, 1.0);
}

#[test]
fn learn_xor_relu() {
    let label_index = 2;
    let input_nodes = input_node_batch(label_index);
    let first_layer = {
        let mut layer = Vec::new();
        for _ in 0..10 {
            let linear_node =
                linear_node(clone_node_batch(&input_nodes), None, None, None, None).unwrap();
            layer.push(linear_node);
        }
        layer
    };
    let first_layer_relu = {
        let mut layer = Vec::new();
        for node in first_layer {
            let relu_node = relu_node(node);
            layer.push(Arc::new(Mutex::new(relu_node)));
        }
        layer
    };
    let second_layer = {
        let mut layer = Vec::new();
        for _ in 0..10 {
            let linear_node =
                linear_node(clone_node_batch(&first_layer_relu), None, None, None, None).unwrap();
            layer.push(linear_node);
        }
        layer
    };
    let second_layer_relu = {
        let mut layer = Vec::new();
        for node in second_layer {
            let relu_node = relu_node(node);
            layer.push(Arc::new(Mutex::new(relu_node)));
        }
        layer
    };
    let linear_output = linear_node(second_layer_relu, None, None, None, None).unwrap();
    let output = sigmoid_node(linear_output);
    let output = Arc::new(Mutex::new(output));
    let label_node = input_node(label_index);
    let error_node = l2_error_node(Arc::clone(&output), Arc::new(Mutex::new(label_node)));
    let step_size = 0.05;
    let mut network = NeuralNetwork::new(
        vec![output],
        Arc::new(Mutex::new(error_node)),
        vec![label_index],
        step_size,
    );

    let dataset = vec![
        vec![0.0, 0.0, 0.0],
        vec![0.0, 1.0, 1.0],
        vec![1.0, 0.0, 1.0],
        vec![1.0, 1.0, 0.0],
    ];

    let max_steps = 5_000;
    network.train(&dataset, max_steps, TrainOption::StochasticGradientDescent);
    for inputs in &dataset {
        let ret = network.evaluate(inputs);
        assert!((ret[0] - inputs[label_index]).abs() < 0.1);
    }
    let ret = network.accuracy(&dataset, binary_accurate);
    assert_eq!(ret, 1.0);
}

fn binary_accurate(eval: Vec<f64>, label: Vec<f64>) -> bool {
    (eval[0] - label[0]).abs() < 0.5
}
