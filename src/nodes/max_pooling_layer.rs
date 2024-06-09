use std::sync::{Arc, Mutex};

use crate::{
    node::Node,
    tensor::{NonZeroShape, OwnedShape, Stride, Tensor},
};

use super::{
    kernel_layer::{kernel_layer, KernelParams},
    max_node::max_node,
};

pub fn max_pooling_layer(
    inputs: Tensor<'_, Arc<Mutex<Node>>>,
    stride: &Stride,
    kernel_shape: &NonZeroShape,
) -> (Vec<Arc<Mutex<Node>>>, OwnedShape) {
    let create_filter = |params: KernelParams| -> Arc<Mutex<Node>> {
        Arc::new(Mutex::new(max_node(params.inputs)))
    };
    kernel_layer(inputs, stride, kernel_shape, create_filter)
}

#[cfg(test)]
mod tests {
    use std::num::NonZeroUsize;

    use crate::{
        nodes::input_node::{input_node_batch, InputNodeBatchParams},
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
        let (max_pooling_layer, layer_shape) = max_pooling_layer(inputs, &stride, &kernel_shape);
        assert_eq!(layer_shape, [2, 2]);
        let mut outputs = vec![];
        for output_node in &max_pooling_layer {
            let mut output_node = output_node.lock().unwrap();
            let output = output_node.evaluate_once(&image, 0);
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
