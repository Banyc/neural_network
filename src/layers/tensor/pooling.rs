use graph::NodeIdx;

use crate::{
    layers::kernel::{kernel_layer, KernelLayerConfig, KernelParams},
    node::GraphBuilder,
    nodes::{max::max_node, mean::mean_node},
    tensor::{OwnedShape, Tensor},
};

pub fn max_pooling_layer(
    graph: &mut GraphBuilder,
    inputs: Tensor<'_, NodeIdx>,
    config: KernelLayerConfig<'_>,
) -> (Vec<NodeIdx>, OwnedShape) {
    let create_filter =
        |params: KernelParams| -> NodeIdx { graph.insert_node(max_node(params.inputs)) };
    kernel_layer(inputs, config, create_filter)
}

pub fn avg_pooling_layer(
    graph: &mut GraphBuilder,
    inputs: Tensor<'_, NodeIdx>,
    config: KernelLayerConfig<'_>,
) -> (Vec<NodeIdx>, OwnedShape) {
    let create_filter =
        |params: KernelParams| -> NodeIdx { graph.insert_node(mean_node(params.inputs)) };
    kernel_layer(inputs, config, create_filter)
}

#[cfg(test)]
mod tests {
    use std::num::NonZeroUsize;

    use graph::dependency_order;

    use crate::{
        computation::ComputationMode,
        node::{evaluate_once, NodeContext},
        nodes::input::{input_node_batch, InputNodeBatchParams},
        param::Params,
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
        let mut graph = GraphBuilder::new();
        let input_nodes = graph.insert_nodes(input_node_batch(InputNodeBatchParams {
            start: 0,
            len: image.len(),
        }));
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
        let (max_pooling_layer, _layer_shape) =
            max_pooling_layer(&mut graph, inputs, kernel_layer_config);
        let mut graph = graph.build();
        let mut outputs = vec![];
        let mut cx = NodeContext::new();
        let nodes_forward = dependency_order(&graph, &max_pooling_layer);
        let mut params = Params::new();
        evaluate_once(
            &mut graph,
            &nodes_forward,
            &mut params,
            &[&image],
            &mut cx,
            ComputationMode::Inference,
        );
        for &output_node in &max_pooling_layer {
            let output_node = graph.nodes().get(output_node).unwrap();
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
