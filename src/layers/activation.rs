use graph::NodeIdx;

use crate::{
    node::CompNode,
    nodes::{relu::relu_node, sigmoid::sigmoid_node, swish::swish_node, tanh::tanh_node},
};

#[derive(Debug, Clone)]
pub enum Activation {
    Sigmoid,
    Tanh,
    ReLu,
    Swish,
}
impl Activation {
    pub fn activate(&self, inputs: &[NodeIdx]) -> Vec<CompNode> {
        inputs
            .iter()
            .copied()
            .map(|x| match self {
                Activation::Sigmoid => sigmoid_node(x),
                Activation::Tanh => tanh_node(x),
                Activation::ReLu => relu_node(x),
                Activation::Swish => swish_node(x),
            })
            .collect::<Vec<CompNode>>()
    }
}
