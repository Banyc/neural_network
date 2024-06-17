use std::sync::Arc;

use crate::{
    mut_cell::MutCell,
    node::SharedNode,
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
    pub fn activate(&self, inputs: &[SharedNode]) -> Vec<SharedNode> {
        inputs
            .iter()
            .map(Arc::clone)
            .map(|x| match self {
                Activation::Sigmoid => sigmoid_node(x),
                Activation::Tanh => tanh_node(x),
                Activation::ReLu => relu_node(x),
                Activation::Swish => swish_node(x),
            })
            .map(|x| Arc::new(MutCell::new(x)))
            .collect::<Vec<SharedNode>>()
    }
}
