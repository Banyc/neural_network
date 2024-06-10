use std::{
    io::Read,
    num::NonZeroUsize,
    path::Path,
    sync::{Arc, Mutex},
};

use strict_num::FiniteF64;

use crate::{
    layers::{conv_relu_max_pooling_layer, dense_relu_layer},
    neural_network::{AccurateFnParams, EvalOption, NeuralNetwork, TrainOption},
    node::SharedNode,
    nodes::{
        conv::{ConvLayerConfig, DeepConvLayerConfig},
        input::{input_node_batch, InputNodeBatchParams},
        kernel::KernelLayerConfig,
        linear::LinearLayerConfig,
        mse::mse_node,
    },
    param::{
        tests::{param_injector, save_params},
        ParamInjection,
    },
    tensor::{primitive_to_stride, shape_to_non_zero, Tensor},
};

const CLASSES: usize = 10;
const TRAIN_IMAGE: &str = "local/mnist/train-images.idx3-ubyte";
const TRAIN_LABEL: &str = "local/mnist/train-labels.idx1-ubyte";
const TEST_IMAGE: &str = "local/mnist/t10k-images.idx3-ubyte";
const TEST_LABEL: &str = "local/mnist/t10k-labels.idx1-ubyte";
const PARAMS_BIN: &str = "local/mnist/params.bincode";
const PARAMS_TXT: &str = "local/mnist/params.ron";

#[ignore]
#[test]
fn mnist_converge() {
    let train_dataset = read_mnist(TRAIN_IMAGE, TRAIN_LABEL).unwrap();
    println!("inputs: {:?}", train_dataset[0]);
    let mut nn = neural_network(0.1, None);
    let _acc = nn.accuracy(&train_dataset[0..1], accurate);
    {
        let option = EvalOption::ClearCache;
        let loss = nn.compute_error(&train_dataset[0], option, 0);
        println!("loss: {loss}");
    }
    let max_steps = 128;
    let option = TrainOption::StochasticGradientDescent;
    nn.train(&train_dataset[0..1], max_steps, option);
    {
        let option = EvalOption::ClearCache;
        let loss = nn.compute_error(&train_dataset[0], option, 0);
        println!("loss: {loss}");
    }
    let eval = nn.evaluate(&train_dataset[0]);
    println!("eval: {eval:?}");
    let acc = nn.accuracy(&train_dataset[0..1], accurate);
    println!("acc: {acc}");
    assert_eq!(acc, 1.);
}

#[ignore]
#[test]
fn mnist() {
    let train_dataset = read_mnist(TRAIN_IMAGE, TRAIN_LABEL).unwrap();
    let test_dataset = read_mnist(TEST_IMAGE, TEST_LABEL).unwrap();
    // epochs
    let mut param_injector = param_injector(PARAMS_BIN);
    let param_injection = ParamInjection {
        injector: &mut param_injector,
        name: "".into(),
    };
    let mut nn = neural_network(0.1, Some(param_injection));
    for i in 0.. {
        println!("epoch: {i}");
        let max_steps = 2 << 10;
        let option = TrainOption::StochasticGradientDescent;
        nn.train(&train_dataset, max_steps, option);
        let acc = nn.accuracy(&test_dataset[..128], accurate);
        println!("acc: {acc}");
        {
            let mut losses = vec![];
            for inputs in &test_dataset[..128] {
                let option = EvalOption::ClearCache;
                let loss = nn.compute_error(inputs, option, 0);
                losses.push(loss);
            }
            let loss = losses
                .iter()
                .copied()
                .map(|x| x / losses.len() as f64)
                .sum::<f64>();
            println!("loss: {loss}");
        }
        save_params(&param_injector.collect_parameters(), PARAMS_BIN, PARAMS_TXT).unwrap();
    }
}

/// a LeNet variant
fn neural_network(
    step_size: f64,
    mut param_injection: Option<ParamInjection<'_>>,
) -> NeuralNetwork {
    let width = 28;
    let height = 28;
    let input_nodes = input_node_batch(InputNodeBatchParams {
        start: 0,
        len: width * height,
    });
    let (layer, shape) = {
        let shape = [width, height];
        let inputs = Tensor::new(&input_nodes, &shape).unwrap();
        let conv = DeepConvLayerConfig {
            depth: NonZeroUsize::new(6).unwrap(),
            conv: ConvLayerConfig {
                kernel_layer: KernelLayerConfig {
                    stride: &primitive_to_stride(&[1, 1]).unwrap(),
                    kernel_shape: &shape_to_non_zero(&[5, 5]).unwrap(),
                    assert_output_shape: None,
                },
                initial_weights: None,
                initial_bias: None,
                lambda: None,
            },
            assert_output_shape: Some(&[24, 24, 6]),
        };
        let max_pooling = KernelLayerConfig {
            stride: &primitive_to_stride(&[2, 2, 1]).unwrap(),
            kernel_shape: &shape_to_non_zero(&[2, 2, 1]).unwrap(),
            assert_output_shape: Some(&[12, 12, 6]),
        };
        let param_injection = param_injection.as_mut().map(|x| x.name_append(":conv.0"));
        conv_relu_max_pooling_layer(inputs, conv, max_pooling, param_injection)
    };
    let (layer, _shape) = {
        let inputs = Tensor::new(&layer, &shape).unwrap();
        let conv = DeepConvLayerConfig {
            depth: NonZeroUsize::new(16).unwrap(),
            conv: ConvLayerConfig {
                kernel_layer: KernelLayerConfig {
                    stride: &primitive_to_stride(&[1, 1, 1]).unwrap(),
                    kernel_shape: &shape_to_non_zero(&[5, 5, 6]).unwrap(),
                    assert_output_shape: None,
                },
                initial_weights: None,
                initial_bias: None,
                lambda: None,
            },
            assert_output_shape: Some(&[8, 8, 16]),
        };
        let max_pooling = KernelLayerConfig {
            stride: &primitive_to_stride(&[2, 2, 1]).unwrap(),
            kernel_shape: &shape_to_non_zero(&[2, 2, 1]).unwrap(),
            assert_output_shape: Some(&[4, 4, 16]),
        };
        let param_injection = param_injection.as_mut().map(|x| x.name_append(":conv.1"));
        conv_relu_max_pooling_layer(inputs, conv, max_pooling, param_injection)
    };
    let layer = {
        let config = LinearLayerConfig {
            depth: NonZeroUsize::new(120).unwrap(),
            initial_weights: None,
            initial_bias: None,
            lambda: None,
        };
        let param_injection = param_injection.as_mut().map(|x| x.name_append(":dense.0"));
        dense_relu_layer(layer, config, param_injection)
    };
    let layer = {
        let config = LinearLayerConfig {
            depth: NonZeroUsize::new(84).unwrap(),
            initial_weights: None,
            initial_bias: None,
            lambda: None,
        };
        let param_injection = param_injection.as_mut().map(|x| x.name_append(":dense.1"));
        dense_relu_layer(layer, config, param_injection)
    };
    let outputs = {
        let config = LinearLayerConfig {
            depth: NonZeroUsize::new(CLASSES).unwrap(),
            initial_weights: None,
            initial_bias: None,
            lambda: None,
        };
        let param_injection = param_injection.as_mut().map(|x| x.name_append(":dense.2"));
        dense_relu_layer(layer, config, param_injection)
    };
    let label_nodes = input_node_batch(InputNodeBatchParams {
        start: input_nodes.len(),
        len: CLASSES,
    });
    let error_node_inputs = outputs
        .iter()
        .cloned()
        .chain(label_nodes)
        .collect::<Vec<SharedNode>>();
    let error_node = Arc::new(Mutex::new(mse_node(error_node_inputs)));
    NeuralNetwork::new(outputs, error_node, step_size)
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

fn accurate(params: AccurateFnParams<'_>) -> bool {
    let eval = params.outputs;
    let label = &params.inputs[params.inputs.len() - CLASSES..];
    assert_eq!(eval.len(), label.len());
    assert!(!eval.is_empty());
    let eval_max_i = max_i(&eval);
    let label_max_i = max_i(label);
    eval_max_i == label_max_i
}

fn max_i(x: &[f64]) -> usize {
    assert!(!x.is_empty());
    let mut max = x[0];
    let mut max_i = 0;
    for (i, x) in x.iter().copied().enumerate() {
        FiniteF64::new(x).unwrap();
        if max < x {
            max = x;
            max_i = i;
        }
    }
    max_i
}
