use std::sync::{Arc, Mutex};

use crate::{
    node::Node,
    tensor::{OwnedShape, Shape, Stride, Tensor},
};

use super::{
    kernel_layer::{kernel_layer, KernelParams},
    max_node::max_node,
};

pub fn max_pooling_layer(
    inputs: Tensor<'_, Arc<Mutex<Node>>>,
    stride: &Stride,
    kernel_shape: &Shape,
) -> (Vec<Arc<Mutex<Node>>>, OwnedShape) {
    let create_filter = |params: KernelParams| -> Arc<Mutex<Node>> {
        Arc::new(Mutex::new(max_node(params.inputs)))
    };
    kernel_layer(inputs, stride, kernel_shape, create_filter)
}
