use graph::NodeIdx;

use crate::{
    mut_cell::MutCell,
    neural_network::{AccurateFnParams, NeuralNetwork, TrainOption},
    node::{CompNode, GraphBuilder},
    nodes::{
        bias::bias_node,
        input::{input_node, InputNodeGen},
        l2_error::l2_error_node,
        linear::{linear_node, linear_node_manual},
        relu::relu_node,
        sigmoid::sigmoid_node,
        weights::weight_node,
    },
    param::{ParamInjection, ParamInjector, SharedParams},
    ref_ctr::RefCtr,
};

fn single_linear_relu(
    graph: &mut GraphBuilder,
    input_nodes: Vec<NodeIdx>,
    initial_weights: SharedParams,
    initial_bias: SharedParams,
) -> CompNode {
    let linear_node =
        linear_node_manual(graph, input_nodes, None, initial_weights, initial_bias).unwrap();
    relu_node(linear_node)
}

fn single_linear_relu_network(
    node_count: usize,
    initial_weights: SharedParams,
    initial_bias: SharedParams,
) -> NeuralNetwork {
    let mut graph = GraphBuilder::new();
    let mut input_gen = InputNodeGen::new();
    let input_nodes = graph.insert_nodes(input_gen.gen(node_count));
    let relu_node = single_linear_relu(&mut graph, input_nodes, initial_weights, initial_bias);
    let relu_node = graph.insert_node(relu_node);
    let label_node = graph.insert_nodes(input_gen.gen(1))[0];
    let error_node = graph.insert_node(l2_error_node(relu_node, label_node));
    let graph = graph.build();
    NeuralNetwork::new(graph, vec![relu_node], error_node)
}

#[test]
fn evaluate() {
    let initial_weights = vec![3.0, 2.0, 1.0];
    let initial_weights = RefCtr::new(MutCell::new(initial_weights));
    let initial_bias = -20.0;
    let initial_bias = RefCtr::new(MutCell::new(vec![initial_bias]));
    let mut network = single_linear_relu_network(3, initial_weights, initial_bias);
    let ret = network.evaluate(&[&[1.0, 2.0, 3.0]]);
    assert_eq!(ret[0][0], 0.0);
}

#[test]
fn error() {
    let mut graph = GraphBuilder::new();
    let input = graph.insert_node(input_node(0));
    let relu = graph.insert_node(relu_node(input));
    let label = graph.insert_node(input_node(1));
    let error = graph.insert_node(l2_error_node(relu, label));
    let graph = graph.build();
    let mut network = NeuralNetwork::new(graph, vec![relu], error);
    let inputs = vec![-2.0, 1.0];
    let ret = network.evaluate(&[&inputs]);
    assert_eq!(ret[0][0], 0.0);
    let ret = network.error(&[&inputs]);
    assert_eq!(ret, 1.0);
}

#[test]
fn cache_reset() {
    let initial_weights = vec![2.0, 1.0];
    let initial_weights = RefCtr::new(MutCell::new(initial_weights));
    let initial_bias = 3.0;
    let initial_bias = RefCtr::new(MutCell::new(vec![initial_bias]));
    let mut network = single_linear_relu_network(2, initial_weights, initial_bias);
    let ret = network.evaluate(&[&[2.0, -2.0]]);
    assert_eq!(ret[0][0], 5.0);
    let ret = network.evaluate(&[&[6.0, -2.0]]);
    assert!(ret[0][0] != 5.0);
}

#[test]
fn errors_on_dataset() {
    let initial_weights = vec![2.0, 1.0];
    let initial_weights = RefCtr::new(MutCell::new(initial_weights));
    let initial_bias = 3.0;
    let initial_bias = RefCtr::new(MutCell::new(vec![initial_bias]));
    let mut network = single_linear_relu_network(2, initial_weights, initial_bias);
    let dataset = vec![vec![2.0, -2.0, 5.0], vec![6.0, -2.0, 5.0]];
    let ret = network.accuracy(&dataset, binary_accurate);
    assert!(ret > 0.499);
    assert!(ret < 0.501);
}

#[test]
fn gradients() {
    let mut graph = GraphBuilder::new();
    let mut input_gen = InputNodeGen::new();
    let input_nodes = graph.insert_nodes(input_gen.gen(2));
    let initial_weights = vec![2.0, 1.0];
    let initial_weights = RefCtr::new(MutCell::new(initial_weights));
    let weight_node = graph.insert_node(weight_node(input_nodes, initial_weights, None).unwrap());
    let initial_bias = 3.0;
    let initial_bias = RefCtr::new(MutCell::new(vec![initial_bias]));
    let bias_node = graph.insert_node(bias_node(weight_node, initial_bias));
    let relu_node = graph.insert_node(relu_node(bias_node));
    let label_node = graph.insert_nodes(input_gen.gen(1))[0];
    let error_node = graph.insert_node(l2_error_node(relu_node, label_node));
    let graph = graph.build();
    let mut network = NeuralNetwork::new(graph, vec![relu_node], error_node);

    let inputs = vec![2.0, -2.0, 1.0];

    let ret = network.evaluate(&[&inputs]);
    assert_eq!(ret[0][0], 5.0);
    let ret = network.error(&[&inputs]);
    assert_eq!(ret, 16.0);
}

#[test]
fn backpropagate() {
    let mut graph = GraphBuilder::new();
    let mut input_gen = InputNodeGen::new();
    let input_nodes = graph.insert_nodes(input_gen.gen(2));
    let initial_weights = vec![2.0, 1.0];
    let initial_weights = RefCtr::new(MutCell::new(initial_weights));
    let weight_node = graph.insert_node(weight_node(input_nodes, initial_weights, None).unwrap());
    let initial_bias = 3.0;
    let initial_bias = RefCtr::new(MutCell::new(vec![initial_bias]));
    let bias_node = graph.insert_node(bias_node(weight_node, initial_bias));
    let relu_node = graph.insert_node(relu_node(bias_node));
    let label_node = graph.insert_nodes(input_gen.gen(1))[0];
    let error_node = graph.insert_node(l2_error_node(relu_node, label_node));
    let graph = graph.build();
    let step_size = 0.5;
    let mut network = NeuralNetwork::new(graph, vec![relu_node], error_node);

    let inputs = vec![2.0, -2.0, 1.0];
    network.compute_error_and_backpropagate(&[&inputs], step_size);
    {
        let weight_node = network.graph().nodes().get(weight_node).unwrap();
        let weights = weight_node.parameters().borrow();
        assert_eq!(*weights, &[-6.0, 9.0])
    }
    {
        let bias_node = network.graph().nodes().get(bias_node).unwrap();
        let bias = bias_node.parameters().borrow();
        assert_eq!(*bias, &[-1.0])
    }
}

#[test]
fn backpropagate_2() {
    let mut graph = GraphBuilder::new();
    let mut input_gen = InputNodeGen::new();
    let input_nodes = graph.insert_nodes(input_gen.gen(1));
    let initial_weights1 = vec![2.0];
    let initial_weights1 = RefCtr::new(MutCell::new(initial_weights1));
    let weight_node1 = graph.insert_node(weight_node(input_nodes, initial_weights1, None).unwrap());
    let initial_weights2 = vec![3.0];
    let initial_weights2 = RefCtr::new(MutCell::new(initial_weights2));
    let weight_node2 =
        graph.insert_node(weight_node(vec![weight_node1], initial_weights2, None).unwrap());
    let label_node = graph.insert_nodes(input_gen.gen(1))[0];
    let error_node = graph.insert_node(l2_error_node(weight_node2, label_node));
    let graph = graph.build();
    let step_size = 0.5;
    let mut network = NeuralNetwork::new(graph, vec![weight_node2], error_node);

    let inputs = vec![2.0, 1.0];
    network.compute_error_and_backpropagate(&[&inputs], step_size);
    {
        let weight_node = network.graph().nodes().get(weight_node2).unwrap();
        let weights = weight_node.parameters().borrow();
        assert_eq!(*weights, &[-41.0]); // 3 - 0.5 * 88
    }
    {
        let weight_node = network.graph().nodes().get(weight_node1).unwrap();
        let weights = weight_node.parameters().borrow();
        assert_eq!(*weights, &[-64.0]); // 2 - 0.5 * 121
    }
}

#[test]
fn learn_xor_sigmoid() {
    let mut param_injector = ParamInjector::empty();
    let mut param_injection = ParamInjection {
        injector: &mut param_injector,
        name: "".to_string(),
    };

    let mut graph = GraphBuilder::new();
    let mut input_gen = InputNodeGen::new();
    let input_nodes = graph.insert_nodes(input_gen.gen(2));
    let linear_node_1 = {
        let param_injection = param_injection.name_append(":linear.0");
        linear_node(&mut graph, input_nodes.clone(), None, param_injection).unwrap()
    };
    let linear_node_2 = {
        let param_injection = param_injection.name_append(":linear.1");
        linear_node(&mut graph, input_nodes.clone(), None, param_injection).unwrap()
    };
    let linear_node_3 = {
        let param_injection = param_injection.name_append(":linear.2");
        linear_node(&mut graph, input_nodes.clone(), None, param_injection).unwrap()
    };
    let sigmoid_node_1 = graph.insert_node(sigmoid_node(linear_node_1));
    let sigmoid_node_2 = graph.insert_node(sigmoid_node(linear_node_2));
    let sigmoid_node_3 = graph.insert_node(sigmoid_node(linear_node_3));
    let sigmoid_nodes = vec![sigmoid_node_1, sigmoid_node_2, sigmoid_node_3];
    let linear_output = {
        let param_injection = param_injection.name_append(":linear.output");
        linear_node(&mut graph, sigmoid_nodes, None, param_injection).unwrap()
    };
    let output = graph.insert_node(sigmoid_node(linear_output));
    let label_node = graph.insert_nodes(input_gen.gen(1))[0];
    let error_node = graph.insert_node(l2_error_node(output, label_node));
    let graph = graph.build();
    let step_size = 0.5;
    let mut network = NeuralNetwork::new(graph, vec![output], error_node);

    let dataset = vec![
        vec![0.0, 0.0, 0.0],
        vec![0.0, 1.0, 1.0],
        vec![1.0, 0.0, 1.0],
        vec![1.0, 1.0, 0.0],
    ];

    let max_steps = 10_000;
    network.train(
        &dataset,
        step_size,
        max_steps,
        TrainOption::StochasticGradientDescent,
    );
    for inputs in &dataset {
        let ret = network.evaluate(&[inputs]);
        assert!((ret[0][0] - inputs.last().unwrap()).abs() < 0.1);
    }
    let ret = network.accuracy(&dataset, binary_accurate);
    assert_eq!(ret, 1.0);
}

#[test]
fn learn_xor_regularized_sigmoid() {
    let mut param_injector = ParamInjector::empty();
    let mut param_injection = ParamInjection {
        injector: &mut param_injector,
        name: "".to_string(),
    };

    let lambda = 0.0001;
    let mut graph = GraphBuilder::new();
    let mut input_gen = InputNodeGen::new();
    let input_nodes = graph.insert_nodes(input_gen.gen(2));
    let linear_node_1 = {
        let param_injection = param_injection.name_append(":linear.0");
        linear_node(
            &mut graph,
            input_nodes.clone(),
            Some(lambda),
            param_injection,
        )
        .unwrap()
    };
    let linear_node_2 = {
        let param_injection = param_injection.name_append(":linear.1");
        linear_node(
            &mut graph,
            input_nodes.clone(),
            Some(lambda),
            param_injection,
        )
        .unwrap()
    };
    let linear_node_3 = {
        let param_injection = param_injection.name_append(":linear.2");
        linear_node(
            &mut graph,
            input_nodes.clone(),
            Some(lambda),
            param_injection,
        )
        .unwrap()
    };
    let sigmoid_node_1 = graph.insert_node(sigmoid_node(linear_node_1));
    let sigmoid_node_2 = graph.insert_node(sigmoid_node(linear_node_2));
    let sigmoid_node_3 = graph.insert_node(sigmoid_node(linear_node_3));
    let sigmoid_nodes = vec![sigmoid_node_1, sigmoid_node_2, sigmoid_node_3];
    let linear_output = {
        let param_injection = param_injection.name_append(":linear.output");
        linear_node(&mut graph, sigmoid_nodes, Some(lambda), param_injection).unwrap()
    };
    let output = graph.insert_node(sigmoid_node(linear_output));
    let label_node = graph.insert_nodes(input_gen.gen(1))[0];
    let error_node = graph.insert_node(l2_error_node(output, label_node));
    let graph = graph.build();
    let step_size = 0.5;
    let mut network = NeuralNetwork::new(graph, vec![output], error_node);

    let dataset = vec![
        vec![-1.0, -1.0, 0.0],
        vec![-1.0, 1.0, 1.0],
        vec![1.0, -1.0, 1.0],
        vec![1.0, 1.0, 0.0],
    ];

    let max_steps = 10_000;
    network.train(
        &dataset,
        step_size,
        max_steps,
        TrainOption::StochasticGradientDescent,
    );
    for inputs in &dataset {
        let ret = network.evaluate(&[inputs]);
        assert!((ret[0][0] - inputs.last().unwrap()).abs() < 0.1);
    }
    let ret = network.accuracy(&dataset, binary_accurate);
    assert_eq!(ret, 1.0);
}

#[test]
fn learn_xor_relu() {
    let mut param_injector = ParamInjector::empty();
    let mut param_injection = ParamInjection {
        injector: &mut param_injector,
        name: "".to_string(),
    };

    let mut graph = GraphBuilder::new();
    let mut input_gen = InputNodeGen::new();
    let input_nodes = graph.insert_nodes(input_gen.gen(2));
    let first_layer = {
        let mut layer = Vec::new();
        let mut param_injection = param_injection.name_append(":layer.0");
        for i in 0..10 {
            let param_injection = param_injection.name_append(&format!(":linear.{i}"));
            let linear_node =
                linear_node(&mut graph, input_nodes.clone(), None, param_injection).unwrap();
            layer.push(linear_node);
        }
        layer
    };
    let first_layer_relu = {
        let mut layer = Vec::new();
        for node in first_layer {
            let relu_node = graph.insert_node(relu_node(node));
            layer.push(relu_node);
        }
        layer
    };
    let second_layer = {
        let mut layer = Vec::new();
        let mut param_injection = param_injection.name_append(":layer.1");
        for i in 0..10 {
            let param_injection = param_injection.name_append(&format!(":linear.{i}"));
            let linear_node =
                linear_node(&mut graph, first_layer_relu.clone(), None, param_injection).unwrap();
            layer.push(linear_node);
        }
        layer
    };
    let second_layer_relu = {
        let mut layer = Vec::new();
        for node in second_layer {
            let relu_node = graph.insert_node(relu_node(node));
            layer.push(relu_node);
        }
        layer
    };
    let linear_output = {
        let param_injection = param_injection.name_append(":linear.output");
        linear_node(&mut graph, second_layer_relu, None, param_injection).unwrap()
    };
    let output = graph.insert_node(sigmoid_node(linear_output));
    let label_node = graph.insert_nodes(input_gen.gen(1))[0];
    let error_node = graph.insert_node(l2_error_node(output, label_node));
    let graph = graph.build();
    let step_size = 0.05;
    let mut network = NeuralNetwork::new(graph, vec![output], error_node);

    let dataset = vec![
        vec![0.0, 0.0, 0.0],
        vec![0.0, 1.0, 1.0],
        vec![1.0, 0.0, 1.0],
        vec![1.0, 1.0, 0.0],
    ];

    let max_steps = 5_000;
    network.train(
        &dataset,
        step_size,
        max_steps,
        TrainOption::StochasticGradientDescent,
    );
    for inputs in &dataset {
        let ret = network.evaluate(&[inputs]);
        assert!((ret[0][0] - inputs.last().unwrap()).abs() < 0.1);
    }
    let ret = network.accuracy(&dataset, binary_accurate);
    assert_eq!(ret, 1.0);
}

fn binary_accurate(params: AccurateFnParams<'_>) -> bool {
    assert_eq!(params.outputs.len(), 1);
    let eval = params.outputs[0];
    let label = params.inputs.last().unwrap();
    (eval - label).abs() < 0.5
}
