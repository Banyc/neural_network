use std::sync::Arc;

use crate::{
    computation::{NodeBackpropagationComputation, NodeComputation, NodeScalarComputation},
    mut_cell::MutCell,
    node::{Node, SharedNode},
    nodes::{mean::mean_node, std_dev::std_dev_node},
    param::empty_shared_params,
};

/// ```math
/// f(x) = \frac{x - \mu}{\sigma}
/// ```
pub fn layer_norm_layer(operands: Vec<SharedNode>) -> Vec<SharedNode> {
    let mean = mean_node(operands.clone());
    let mean = Arc::new(MutCell::new(mean));
    let mut std_dev_inputs = vec![];
    std_dev_inputs.push(Arc::clone(&mean));
    std_dev_inputs.extend(operands.iter().cloned());
    let std_dev = std_dev_node(std_dev_inputs);
    let std_dev = Arc::new(MutCell::new(std_dev));
    let mut layer_norm_nodes = vec![];
    for operand in operands {
        let computation = LayerNormNodeComputation {};
        let operands = vec![Arc::clone(&mean), Arc::clone(&std_dev), operand];
        let node = Node::new(
            operands,
            Arc::new(MutCell::new(NodeComputation::Scalar(Box::new(computation)))),
            empty_shared_params(),
        );
        layer_norm_nodes.push(Arc::new(MutCell::new(node)));
    }
    layer_norm_nodes
}

#[derive(Debug)]
struct LayerNormNodeComputation {}
impl NodeScalarComputation for LayerNormNodeComputation {
    fn compute_output(
        &self,
        parameters: &[f64],
        operand_outputs: &[f64],
        _graph_inputs: &[f64],
    ) -> f64 {
        assert!(parameters.is_empty());
        assert_eq!(operand_outputs.len(), 3);
        let mean = operand_outputs[0];
        let std_dev = operand_outputs[1];
        let x = operand_outputs[2];
        norm(mean, std_dev, x)
    }
}
impl NodeBackpropagationComputation for LayerNormNodeComputation {
    fn compute_gradient_of_this_at_operand(
        &self,
        parameters: &[f64],
        operand_outputs: &[f64],
        buf: Vec<f64>,
    ) -> Vec<f64> {
        assert!(parameters.is_empty());
        assert_eq!(operand_outputs.len(), 3);
        let mean = operand_outputs[0];
        let std_dev = operand_outputs[1];
        let x = operand_outputs[2];
        norm_derivative(mean, std_dev, x, buf)
    }

    fn compute_gradient_of_this_at_parameter(
        &self,
        parameters: &[f64],
        operand_outputs: &[f64],
        buf: Vec<f64>,
    ) -> Vec<f64> {
        assert!(parameters.is_empty());
        assert_eq!(operand_outputs.len(), 3);
        buf
    }
}

fn norm(mean: f64, std_dev: f64, x: f64) -> f64 {
    if std_dev == 0. {
        panic!();
    }
    (x - mean) / std_dev
}

fn norm_derivative(mean: f64, std_dev: f64, x: f64, mut buf: Vec<f64>) -> Vec<f64> {
    if std_dev == 0. {
        panic!();
    }
    let mean_der = -1. / std_dev;
    let std_dev_der = -(x - mean) / std_dev.powi(2);
    let x_der = 1. / std_dev;
    buf.extend([mean_der, std_dev_der, x_der]);
    buf
}
