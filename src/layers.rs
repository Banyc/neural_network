use std::{cell::RefCell, sync::Arc};

use crate::{
    node::SharedNode,
    nodes::{
        conv::{deep_conv_layer, DeepConvLayerConfig},
        kernel::KernelLayerConfig,
        linear::{linear_layer, LinearLayerConfig},
        max_pooling::max_pooling_layer,
        relu::relu_node,
    },
    param::ParamInjection,
    tensor::{OwnedShape, Tensor},
};

pub fn conv_relu_max_pooling_layer(
    inputs: Tensor<'_, SharedNode>,
    conv: DeepConvLayerConfig<'_>,
    max_pooling: KernelLayerConfig<'_>,
    param_injection: Option<ParamInjection<'_>>,
) -> (Vec<SharedNode>, OwnedShape) {
    let (conv_layer, shape) = deep_conv_layer(inputs, conv, param_injection);
    let relu_layer = conv_layer
        .into_iter()
        .map(|x| Arc::new(RefCell::new(relu_node(x))))
        .collect::<Vec<SharedNode>>();
    {
        let inputs = Tensor::new(&relu_layer, &shape).unwrap();
        max_pooling_layer(inputs, max_pooling)
    }
}

pub fn dense_relu_layer(
    inputs: Vec<SharedNode>,
    config: LinearLayerConfig,
    param_injection: Option<ParamInjection<'_>>,
) -> Vec<SharedNode> {
    let depth = config.depth;
    let linear_layer = linear_layer(inputs, config, param_injection).unwrap();
    assert_eq!(linear_layer.len(), depth.get());
    linear_layer
        .into_iter()
        .map(|x| Arc::new(RefCell::new(relu_node(x))))
        .collect::<Vec<SharedNode>>()
}
