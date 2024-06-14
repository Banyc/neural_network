use std::sync::Arc;

use crate::{
    layers::kernel::{kernel_layer, KernelLayerConfig, KernelParams},
    mut_cell::MutCell,
    node::SharedNode,
    nodes::max::max_node,
    tensor::{OwnedShape, Tensor},
};

pub fn max_pooling_layer(
    inputs: Tensor<'_, SharedNode>,
    config: KernelLayerConfig<'_>,
) -> (Vec<SharedNode>, OwnedShape) {
    let create_filter =
        |params: KernelParams| -> SharedNode { Arc::new(MutCell::new(max_node(params.inputs))) };
    kernel_layer(inputs, config, create_filter)
}

#[cfg(test)]
mod tests {
    use std::num::NonZeroUsize;

    use crate::{
        computation::ComputationMode,
        node::NodeContext,
        nodes::input::{input_node_batch, InputNodeBatchParams},
        tensor::{OwnedNonZeroShape, OwnedStride},
    };

    use super::*;

    #[test]
    fn test_max_pooling() {
        let image = [
            2, 2, 7, 3, //
            9, 4, 6, 1, //
            8, 5, 2, 4, //
            3, 1, 2, 6, //
        ];
        let image = image
            .iter()
            .copied()
            .map(|x| x as f64)
            .collect::<Vec<f64>>();
        let input_nodes = input_node_batch(InputNodeBatchParams {
            start: 0,
            len: image.len(),
        });
        let input_shape = [4, 4];
        let inputs = Tensor::new(&input_nodes, &input_shape).unwrap();
        let stride = [2, 2];
        let stride = stride
            .into_iter()
            .map(|x| NonZeroUsize::new(x).unwrap())
            .collect::<OwnedStride>();
        let kernel_shape = [2, 2];
        let kernel_shape = kernel_shape
            .into_iter()
            .map(|x| NonZeroUsize::new(x).unwrap())
            .collect::<OwnedNonZeroShape>();
        let kernel_layer_config = KernelLayerConfig {
            stride: &stride,
            kernel_shape: &kernel_shape,
            assert_output_shape: Some(&[2, 2]),
        };
        let (max_pooling_layer, _layer_shape) = max_pooling_layer(inputs, kernel_layer_config);
        let mut outputs = vec![];
        let mut cx = NodeContext::new();
        for output_node in &max_pooling_layer {
            let mut output_node = output_node.borrow_mut();
            output_node.evaluate_once(&[&image], &mut cx, ComputationMode::Inference);
            let output = output_node.output().unwrap()[0];
            outputs.push(output);
        }
        assert_eq!(
            outputs,
            [
                9., 7., //
                8., 6., //
            ]
        );
    }
}
