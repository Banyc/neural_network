use std::{
    io::Read,
    num::NonZeroUsize,
    path::Path,
    sync::{Arc, Mutex},
};

use crate::{
    neural_network::{NeuralNetwork, TrainOption},
    node::Node,
    nodes::{
        conv_layer::{self, deep_conv_layer},
        input_node::input_node_batch,
        linear_node, max_pooling_layer,
        mse_node::mse_node,
        sigmoid_node::sigmoid_node,
    },
    tensor::{append_tensors, non_zero_to_shape, primitive_to_stride, shape_to_non_zero, Tensor},
};

const CLASSES: usize = 10;
const TRAIN_IMAGE: &str = "local/train-images.idx3-ubyte";
const TRAIN_LABEL: &str = "local/train-labels.idx1-ubyte";
const TEST_IMAGE: &str = "local/t10k-images.idx3-ubyte";
const TEST_LABEL: &str = "local/t10k-labels.idx1-ubyte";

#[ignore]
#[test]
fn mnist() {
    let train_dataset = read_mnist(TRAIN_IMAGE, TRAIN_LABEL).unwrap();
    let test_dataset = read_mnist(TEST_IMAGE, TEST_LABEL).unwrap();
    {
        let mut nn = neural_network(0.01);
        let max_steps = 1024;
        let option = TrainOption::StochasticGradientDescent;
        nn.train(&train_dataset[0..1], max_steps, option);
        let acc = nn.accuracy(&train_dataset[0..1], accurate);
        println!("acc: {acc}");
        assert_eq!(acc, 1.);
    }
    let mut nn = neural_network(0.01);
    for i in 0.. {
        println!("epoch: {i}");
        let max_steps = 2 << 10;
        let option = TrainOption::StochasticGradientDescent;
        nn.train(&train_dataset, max_steps, option);
        let acc = nn.accuracy(&test_dataset[..128], accurate);
        println!("acc: {acc}");
    }
}

fn neural_network(step_size: f64) -> NeuralNetwork {
    let width = 28;
    let height = 28;
    let input_nodes = input_node_batch(width * height);
    let (conv_layer, shape) = {
        let shape = [width, height];
        let inputs = Tensor::new(&input_nodes, &shape).unwrap();
        let stride = primitive_to_stride(&[1, 1]).unwrap();
        let kernel_shape = shape_to_non_zero(&[5, 5]).unwrap();
        let kernel = conv_layer::KernelConfig {
            shape: &kernel_shape,
            initial_weights: None,
            initial_bias: None,
            lambda: None,
        };
        let depth = NonZeroUsize::new(6).unwrap();
        let (layers, shape) = deep_conv_layer(inputs, &stride, kernel, depth, None);
        append_tensors(layers, &shape)
    };
    let shape = non_zero_to_shape(&shape);
    assert_eq!(shape, [24, 24, 6]);
    let sigmoid_layer = conv_layer
        .into_iter()
        .map(|x| Arc::new(Mutex::new(sigmoid_node(x))))
        .collect::<Vec<Arc<Mutex<Node>>>>();
    let (max_pooling_layer, shape) = {
        let inputs = Tensor::new(&sigmoid_layer, &shape).unwrap();
        let stride = primitive_to_stride(&[2, 2, 1]).unwrap();
        let kernel_shape = shape_to_non_zero(&[2, 2, 1]).unwrap();
        max_pooling_layer::max_pooling_layer(inputs, &stride, &kernel_shape)
    };
    assert_eq!(shape, [12, 12, 6]);
    let (conv_layer, shape) = {
        let inputs = Tensor::new(&max_pooling_layer, &shape).unwrap();
        let stride = primitive_to_stride(&[1, 1, 1]).unwrap();
        let kernel_shape = shape_to_non_zero(&[5, 5, 6]).unwrap();
        let kernel = conv_layer::KernelConfig {
            shape: &kernel_shape,
            initial_weights: None,
            initial_bias: None,
            lambda: None,
        };
        let depth = NonZeroUsize::new(16).unwrap();
        let (layers, shape) = deep_conv_layer(inputs, &stride, kernel, depth, None);
        append_tensors(layers, &shape)
    };
    let shape = non_zero_to_shape(&shape);
    assert_eq!(shape, [8, 8, 16]);
    let sigmoid_layer = conv_layer
        .into_iter()
        .map(|x| Arc::new(Mutex::new(sigmoid_node(x))))
        .collect::<Vec<Arc<Mutex<Node>>>>();
    let (max_pooling_layer, shape) = {
        let inputs = Tensor::new(&sigmoid_layer, &shape).unwrap();
        let stride = primitive_to_stride(&[2, 2, 1]).unwrap();
        let kernel_shape = shape_to_non_zero(&[2, 2, 1]).unwrap();
        max_pooling_layer::max_pooling_layer(inputs, &stride, &kernel_shape)
    };
    assert_eq!(shape, [4, 4, 16]);
    let linear_layer = {
        let depth = NonZeroUsize::new(120).unwrap();
        linear_node::linear_layer(max_pooling_layer, depth, None, None, None, None).unwrap()
    };
    assert_eq!(linear_layer.len(), 120);
    let sigmoid_layer = linear_layer
        .into_iter()
        .map(|x| Arc::new(Mutex::new(sigmoid_node(x))))
        .collect::<Vec<Arc<Mutex<Node>>>>();
    let linear_layer = {
        let depth = NonZeroUsize::new(84).unwrap();
        linear_node::linear_layer(sigmoid_layer, depth, None, None, None, None).unwrap()
    };
    assert_eq!(linear_layer.len(), 84);
    let sigmoid_layer = linear_layer
        .into_iter()
        .map(|x| Arc::new(Mutex::new(sigmoid_node(x))))
        .collect::<Vec<Arc<Mutex<Node>>>>();
    let linear_layer = {
        let depth = NonZeroUsize::new(CLASSES).unwrap();
        linear_node::linear_layer(sigmoid_layer, depth, None, None, None, None).unwrap()
    };
    assert_eq!(linear_layer.len(), CLASSES);
    let label_nodes = input_node_batch(CLASSES);
    let error_node_inputs = linear_layer
        .iter()
        .cloned()
        .chain(label_nodes)
        .collect::<Vec<Arc<Mutex<Node>>>>();
    let error_node = Arc::new(Mutex::new(mse_node(error_node_inputs)));
    let label_indices = (CLASSES..(CLASSES * 2)).collect();
    NeuralNetwork::new(linear_layer, error_node, label_indices, step_size)
}

fn read_mnist(image: impl AsRef<Path>, label: impl AsRef<Path>) -> std::io::Result<Vec<Vec<f64>>> {
    let labels = {
        let file = std::fs::File::options().read(true).open(label)?;
        let mut rdr = std::io::BufReader::new(file);
        let mut magic = [0; 4];
        rdr.read_exact(&mut magic)?;
        let magic = u32::from_be_bytes(magic);
        if magic != 2049 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                format!("Magic number mismatch, expected 2049, got {magic}"),
            ));
        }
        let mut size = [0; 4];
        rdr.read_exact(&mut size)?;
        let size = u32::from_be_bytes(size);
        let size = usize::try_from(size).map_err(|_| std::io::ErrorKind::InvalidInput)?;
        let mut labels = Vec::with_capacity(size);
        let mut label = [0; 1];
        for _ in 0..size {
            rdr.read_exact(&mut label)?;
            labels.push(u8::from_be_bytes(label));
        }
        labels
    };
    let images = {
        let file = std::fs::File::options().read(true).open(image)?;
        let mut rdr = std::io::BufReader::new(file);
        let mut magic = [0; 4];
        rdr.read_exact(&mut magic)?;
        let magic = u32::from_be_bytes(magic);
        if magic != 2051 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                format!("Magic number mismatch, expected 2051, got {magic}"),
            ));
        }
        let mut size = [0; 4];
        rdr.read_exact(&mut size)?;
        let size = u32::from_be_bytes(size);
        let size = usize::try_from(size).map_err(|_| std::io::ErrorKind::InvalidInput)?;
        let mut images = Vec::with_capacity(size);
        let mut rows = [0; 4];
        rdr.read_exact(&mut rows)?;
        let rows = u32::from_be_bytes(rows);
        let mut cols = [0; 4];
        rdr.read_exact(&mut cols)?;
        let cols = u32::from_be_bytes(cols);
        let len = rows * cols;
        let len = usize::try_from(len).map_err(|_| std::io::ErrorKind::InvalidInput)?;
        for _ in 0..size {
            let mut image = vec![0; len];
            rdr.read_exact(&mut image)?;
            images.push(image);
        }
        images
    };
    let dataset = images
        .into_iter()
        .zip(labels)
        .map(|(image, label)| {
            let image = image.into_iter().map(|x| x as f64 / u8::MAX as f64);
            let label = one_hot(label as usize, CLASSES);
            image.chain(label).collect::<Vec<f64>>()
        })
        .collect::<Vec<Vec<f64>>>();
    Ok(dataset)
}

fn one_hot(i: usize, space_size: usize) -> Vec<f64> {
    assert!(i < space_size);
    let mut vec = vec![0.; space_size];
    vec[i] = 1.;
    vec
}

fn accurate(eval: Vec<f64>, label: Vec<f64>) -> bool {
    assert_eq!(eval.len(), label.len());
    assert!(!eval.is_empty());
    let eval_max_i = max_i(&eval);
    let label_max_i = max_i(&label);
    eval_max_i == label_max_i
}

fn max_i(x: &[f64]) -> usize {
    assert!(!x.is_empty());
    let mut max = x[0];
    let mut max_i = 0;
    for (i, x) in x.iter().copied().enumerate() {
        if max < x {
            max = x;
            max_i = i;
        }
    }
    max_i
}