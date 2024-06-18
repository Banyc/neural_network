use std::{io::Read, num::NonZeroUsize, path::Path, sync::Arc};

use crate::{
    layers::{
        activation::Activation,
        dense::dense_layer,
        kernel::KernelLayerConfig,
        tensor::{
            conv::{ConvLayerConfig, DeepConvLayerConfig},
            conv_max_pooling_layer,
        },
    },
    mut_cell::MutCell,
    neural_network::{NeuralNetwork, TrainOption},
    node::SharedNode,
    nodes::{input::InputNodeGen, linear::LinearLayerConfig, mse::mse_node},
    param::{
        tests::{param_injector, save_params},
        ParamInjection, ParamInjector,
    },
    tensor::{primitive_to_stride, shape_to_non_zero, Tensor},
    tests::multi_class_one_hot_accurate,
};

const CLASSES: usize = 10;
const TRAIN_IMAGE: &str = "local/mnist/train-images.idx3-ubyte";
const TRAIN_LABEL: &str = "local/mnist/train-labels.idx1-ubyte";
const TEST_IMAGE: &str = "local/mnist/t10k-images.idx3-ubyte";
const TEST_LABEL: &str = "local/mnist/t10k-labels.idx1-ubyte";
const PARAMS_BIN: &str = "local/mnist/params.bincode";
const PARAMS_TXT: &str = "local/mnist/params.ron";

#[test]
fn converge() {
    let train_dataset = read_mnist(TRAIN_IMAGE, TRAIN_LABEL).unwrap();
    println!("inputs: {:?}", train_dataset[0]);
    let mut param_injector = ParamInjector::empty();
    let param_injection = ParamInjection {
        injector: &mut param_injector,
        name: "".to_string(),
    };
    let mut nn = neural_network(param_injection);
    let loss = nn.error(&train_dataset[0..1]);
    println!("loss: {loss}");
    let mut step_size = loss;
    let mut prev_loss = loss;
    loop {
        let max_steps = 16;
        let option = TrainOption::StochasticGradientDescent;
        nn.train(&train_dataset[0..1], step_size, max_steps, option);

        let eval = nn.evaluate(&train_dataset[0..1]);
        let eval = eval.iter().map(|x| x[0]).collect::<Vec<f64>>();
        println!("eval: {eval:?}");
        let acc = nn.accuracy(&train_dataset[0..1], |x| {
            multi_class_one_hot_accurate(x, CLASSES)
        });
        if acc == 1. {
            break;
        }
        let loss = nn.error(&train_dataset[0..1]);
        println!("loss: {loss}");
        if (prev_loss - loss).abs() < 0.001 {
            param_injector = ParamInjector::empty();
            let param_injection = ParamInjection {
                injector: &mut param_injector,
                name: "".to_string(),
            };
            nn = neural_network(param_injection);
            continue;
        }
        prev_loss = loss;
        step_size = loss;
    }
}

#[ignore]
#[test]
fn train() {
    let train_dataset = read_mnist(TRAIN_IMAGE, TRAIN_LABEL).unwrap();
    let test_dataset = read_mnist(TEST_IMAGE, TEST_LABEL).unwrap();
    // epochs
    let mut param_injector = param_injector(PARAMS_BIN);
    let param_injection = ParamInjection {
        injector: &mut param_injector,
        name: "".into(),
    };
    let mut nn = neural_network(param_injection);
    let loss = nn.error(&train_dataset[0..1]);
    println!("loss: {loss}");
    let mut step_size = loss;
    for i in 0.. {
        println!("epoch: {i}");
        let max_steps = 2 << 10;
        let option = TrainOption::StochasticGradientDescent;
        nn.train(&train_dataset, step_size, max_steps, option);
        let acc = nn.accuracy(&test_dataset[..128], |x| {
            multi_class_one_hot_accurate(x, CLASSES)
        });
        println!("acc: {acc}");
        let loss = nn.error(&test_dataset[..128]);
        println!("loss: {loss}");
        step_size = loss;
        save_params(&param_injector.collect_parameters(), PARAMS_BIN, PARAMS_TXT).unwrap();
    }
}

/// a LeNet variant
fn neural_network(mut param_injection: ParamInjection<'_>) -> NeuralNetwork {
    let activation = Activation::Swish;
    let width = 28;
    let height = 28;
    let mut input_node_gen = InputNodeGen::new();
    let input_nodes = input_node_gen.gen(width * height);
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
                lambda: None,
            },
            assert_output_shape: Some(&[24, 24, 6]),
        };
        let max_pooling = KernelLayerConfig {
            stride: &primitive_to_stride(&[2, 2, 1]).unwrap(),
            kernel_shape: &shape_to_non_zero(&[2, 2, 1]).unwrap(),
            assert_output_shape: Some(&[12, 12, 6]),
        };
        let param_injection = param_injection.name_append(":conv.0");
        conv_max_pooling_layer(
            inputs,
            conv,
            max_pooling,
            &activation,
            None,
            param_injection,
        )
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
                lambda: None,
            },
            assert_output_shape: Some(&[8, 8, 16]),
        };
        let max_pooling = KernelLayerConfig {
            stride: &primitive_to_stride(&[2, 2, 1]).unwrap(),
            kernel_shape: &shape_to_non_zero(&[2, 2, 1]).unwrap(),
            assert_output_shape: Some(&[4, 4, 16]),
        };
        let param_injection = param_injection.name_append(":conv.1");
        conv_max_pooling_layer(
            inputs,
            conv,
            max_pooling,
            &activation,
            None,
            param_injection,
        )
    };
    let layer = {
        let config = LinearLayerConfig {
            depth: NonZeroUsize::new(120).unwrap(),
            lambda: None,
        };
        let param_injection = param_injection.name_append(":dense.0");
        dense_layer(layer, config, &activation, param_injection)
    };
    let layer = {
        let config = LinearLayerConfig {
            depth: NonZeroUsize::new(84).unwrap(),
            lambda: None,
        };
        let param_injection = param_injection.name_append(":dense.1");
        dense_layer(layer, config, &activation, param_injection)
    };
    let outputs = {
        let config = LinearLayerConfig {
            depth: NonZeroUsize::new(CLASSES).unwrap(),
            lambda: None,
        };
        let param_injection = param_injection.name_append(":dense.2");
        dense_layer(layer, config, &activation, param_injection)
    };
    let label_nodes = input_node_gen.gen(CLASSES);
    let error_node_inputs = label_nodes
        .into_iter()
        .chain(outputs.iter().cloned())
        .collect::<Vec<SharedNode>>();
    let error_node = Arc::new(MutCell::new(mse_node(error_node_inputs)));
    NeuralNetwork::new(outputs, error_node)
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
