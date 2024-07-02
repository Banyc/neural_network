use graph::NodeIdx;

use crate::{
    computation::{NodeBackpropagationComputation, NodeComputation, NodeScalarComputation},
    node::{CompNode, GraphBuilder},
    nodes::{mean::mean_node, std_dev::std_dev_node},
    param::empty_shared_params,
};

/// ```math
/// f(x) = \frac{x - \mu}{\sigma}
/// ```
pub fn layer_norm_layer(graph: &mut GraphBuilder, operands: Vec<NodeIdx>) -> Vec<CompNode> {
    let mean = graph.insert_node(mean_node(operands.clone()));
    let mut std_dev_inputs = vec![];
    std_dev_inputs.push(mean);
    std_dev_inputs.extend(operands.iter().cloned());
    let std_dev = graph.insert_node(std_dev_node(std_dev_inputs));
    let mut layer_norm_nodes = vec![];
    for operand in operands {
        let computation = LayerNormNodeComputation {};
        let operands = vec![mean, std_dev, operand];
        let node = CompNode::new(
            operands,
            NodeComputation::Scalar(Box::new(computation)),
            empty_shared_params(),
        );
        layer_norm_nodes.push(node);
    }
    layer_norm_nodes
}

#[derive(Debug, Clone)]
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
