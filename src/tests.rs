use std::sync::{Arc, Mutex};

use crate::{
    neural_network::NeuralNetwork,
    nodes::{
        bias_node::bias_node,
        input_node::{input_node, input_node_batch},
        l2_error_node::l2_error_node,
        linear_node::linear_node,
        node::{clone_node_batch, GeneralNode},
        relu_node::relu_node,
        sigmoid_node::sigmoid_node,
        weight_node::weight_node,
    },
};

fn single_linear_relu(
    input_nodes: Vec<Arc<Mutex<GeneralNode>>>,
    initial_weights: Option<Vec<f64>>,
    initial_bias: Option<f64>,
) -> GeneralNode {
    let linear_node = linear_node(input_nodes, initial_weights, initial_bias).unwrap();
    let relu_node = relu_node(Arc::new(Mutex::new(linear_node)));
    relu_node
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
    let neural_network = NeuralNetwork::new(
        relu_node,
        Arc::new(Mutex::new(error_node)),
        node_count,
        1e-2,
    );
    neural_network
}

#[test]
fn evaluate() {
    let network = single_linear_relu_network(3, Some(vec![3.0, 2.0, 1.0]), Some(-20.0));
    let ret = network.evaluate_and_reset_caches(&vec![1.0, 2.0, 3.0]);
    assert_eq!(ret, 0.0);
}

#[test]
fn error() {
    let input = input_node(0);
    let relu = relu_node(Arc::new(Mutex::new(input)));
    let relu = Arc::new(Mutex::new(relu));
    let label = input_node(1);
    let error = l2_error_node(Arc::clone(&relu), Arc::new(Mutex::new(label)));
    let error = Arc::new(Mutex::new(error));
    let network = NeuralNetwork::new(relu, error, 1, 1e-2);
    let inputs = vec![-2.0, 1.0];
    let ret = network.evaluate_and_reset_caches(&inputs);
    assert_eq!(ret, 0.0);
    let ret = network.compute_error_and_reset_caches(&inputs);
    assert_eq!(ret, 1.0);
}

#[test]
fn cache_reset() {
    let network = single_linear_relu_network(2, Some(vec![2.0, 1.0]), Some(3.0));
    let ret = network.evaluate_and_reset_caches(&vec![2.0, -2.0]);
    assert_eq!(ret, 5.0);
    let ret = network.evaluate_and_reset_caches(&vec![6.0, -2.0]);
    assert!(ret != 5.0);
}

#[test]
fn errors_on_dataset() {
    let network = single_linear_relu_network(2, Some(vec![2.0, 1.0]), Some(3.0));
    let dataset = vec![vec![2.0, -2.0, 5.0], vec![6.0, -2.0, 5.0]];
    let ret = network.errors_on_dataset(&dataset);
    assert!(ret > 0.499);
    assert!(ret < 0.501);
}

#[test]
fn gradients() {
    let label_index = 2;
    let input_nodes = input_node_batch(label_index);
    let initial_weights = vec![2.0, 1.0];
    let weight_node = weight_node(input_nodes, Some(initial_weights)).unwrap();
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
    let network = NeuralNetwork::new(
        Arc::clone(&relu_node),
        Arc::clone(&error_node),
        label_index,
        1e-2,
    );

    let inputs = vec![2.0, -2.0, 1.0];

    let ret = network.evaluate(&inputs);
    assert_eq!(ret, 5.0);
    {
        let relu_node = relu_node.lock().unwrap();
        assert!(relu_node.output().unwrap() > 0.0);
    }
    let ret = network.compute_error(&inputs);
    assert_eq!(ret, 16.0);

    // fill all caches
    network.backpropagation_step(&inputs);

    {
        let mut error_node = error_node.lock().unwrap();
        assert_eq!(error_node.global_gradient().unwrap(), 1.0);
        assert_eq!(
            error_node.local_operand_gradient().unwrap().as_ref(),
            &vec![8.0, -8.0]
        );
    }

    {
        let mut relu_node = relu_node.lock().unwrap();
        assert_eq!(relu_node.global_gradient().unwrap(), 8.0);
        assert_eq!(
            relu_node.local_operand_gradient().unwrap().as_ref(),
            &vec![1.0]
        );
        assert_eq!(
            relu_node.local_parameter_gradient().unwrap().as_ref(),
            &vec![]
        );
        assert_eq!(
            relu_node.global_parameter_gradient().unwrap().as_ref(),
            &vec![]
        );
    }

    {
        let mut bias_node = bias_node.lock().unwrap();
        assert_eq!(bias_node.global_gradient().unwrap(), 8.0);
        assert_eq!(
            bias_node.local_operand_gradient().unwrap().as_ref(),
            &vec![1.0]
        );
        assert_eq!(
            bias_node.local_parameter_gradient().unwrap().as_ref(),
            &vec![1.0]
        );
        assert_eq!(
            bias_node.global_parameter_gradient().unwrap().as_ref(),
            &vec![8.0]
        );
    }

    {
        let mut weight_node = weight_node.lock().unwrap();
        assert_eq!(weight_node.global_gradient().unwrap(), 8.0);
        assert_eq!(
            weight_node.local_operand_gradient().unwrap().as_ref(),
            &vec![2.0, 1.0]
        );
        assert_eq!(
            weight_node.local_parameter_gradient().unwrap().as_ref(),
            &vec![2.0, -2.0]
        );
        assert_eq!(
            weight_node.global_parameter_gradient().unwrap().as_ref(),
            &vec![16.0, -16.0]
        )
    }
}

#[test]
fn backpropagation_step() {
    let label_index = 2;
    let input_nodes = input_node_batch(label_index);
    let initial_weights = vec![2.0, 1.0];
    let weight_node = weight_node(input_nodes, Some(initial_weights)).unwrap();
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
    let network = NeuralNetwork::new(
        Arc::clone(&relu_node),
        Arc::clone(&error_node),
        label_index,
        step_size,
    );

    let inputs = vec![2.0, -2.0, 1.0];
    network.backpropagation_step(&inputs);

    {
        let weight_node = weight_node.lock().unwrap();
        assert_eq!(weight_node.parameters(), &vec![-6.0, 9.0])
    }

    {
        let bias_node = bias_node.lock().unwrap();
        assert_eq!(bias_node.parameters(), &vec![-1.0])
    }
}

#[test]
fn learn_xor_sigmoid() {
    let label_index = 2;
    let input_nodes = input_node_batch(label_index);
    let linear_node_1 = linear_node(clone_node_batch(&input_nodes), None, None).unwrap();
    let linear_node_2 = linear_node(clone_node_batch(&input_nodes), None, None).unwrap();
    let linear_node_3 = linear_node(clone_node_batch(&input_nodes), None, None).unwrap();
    let sigmoid_node_1 = sigmoid_node(Arc::new(Mutex::new(linear_node_1)));
    let sigmoid_node_2 = sigmoid_node(Arc::new(Mutex::new(linear_node_2)));
    let sigmoid_node_3 = sigmoid_node(Arc::new(Mutex::new(linear_node_3)));
    let sigmoid_nodes = vec![
        Arc::new(Mutex::new(sigmoid_node_1)),
        Arc::new(Mutex::new(sigmoid_node_2)),
        Arc::new(Mutex::new(sigmoid_node_3)),
    ];
    let linear_output = linear_node(sigmoid_nodes, None, None).unwrap();
    let output = sigmoid_node(Arc::new(Mutex::new(linear_output)));
    let output = Arc::new(Mutex::new(output));
    let label_node = input_node(label_index);
    let error_node = l2_error_node(Arc::clone(&output), Arc::new(Mutex::new(label_node)));
    let step_size = 0.5;
    let network = NeuralNetwork::new(
        output,
        Arc::new(Mutex::new(error_node)),
        label_index,
        step_size,
    );

    let dataset = vec![
        vec![0.0, 0.0, 0.0],
        vec![0.0, 1.0, 1.0],
        vec![1.0, 0.0, 1.0],
        vec![1.0, 1.0, 0.0],
    ];

    let max_steps = 10_000;
    network.train(&dataset, max_steps);
    for inputs in &dataset {
        let ret = network.evaluate_and_reset_caches(inputs);
        assert!((ret - inputs[label_index]).abs() < 0.1);
    }
    let ret = network.errors_on_dataset(&dataset);
    assert_eq!(ret, 0.0);
}
