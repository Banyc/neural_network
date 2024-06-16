use std::{num::NonZeroUsize, sync::Arc};

use crate::{
    layers::activation::Activation,
    mut_cell::MutCell,
    node::SharedNode,
    nodes::{
        linear::{linear_layer, LinearLayerConfig},
        recurrent::{recurrent_reader_node, recurrent_writer_node},
        sum::sum_node,
    },
    param::ParamInjection,
};

/// gradient computation algorithm: BPTT but with a sequence length of one
pub fn rnn(
    inputs: Vec<SharedNode>,
    depth: NonZeroUsize,
    activation: &Activation,
    mut param_injection: ParamInjection,
) -> Vec<SharedNode> {
    let mut rec_readers = vec![];
    let mut rec_values = vec![];
    for _ in 0..depth.get() {
        let (reader, value) = recurrent_reader_node();
        rec_readers.push(Arc::new(MutCell::new(reader)));
        rec_values.push(value);
    }
    let x = {
        let param_injection = param_injection.name_append(":input");
        let config = LinearLayerConfig {
            depth,
            lambda: None,
        };
        linear_layer(inputs, config, param_injection).unwrap()
    };
    let rec = {
        let param_injection = param_injection.name_append(":rec");
        let config = LinearLayerConfig {
            depth,
            lambda: None,
        };
        linear_layer(rec_readers, config, param_injection).unwrap()
    };
    let mut sum_layer = vec![];
    for (x, rec) in x.into_iter().zip(rec.into_iter()) {
        let node = sum_node(vec![x, rec]);
        sum_layer.push(Arc::new(MutCell::new(node)));
    }
    let act_layer = activation.activate(&sum_layer);

    assert_eq!(act_layer.len(), rec_values.len());
    let mut rec_writers = vec![];
    for (input, value) in act_layer.into_iter().zip(rec_values.into_iter()) {
        let writer = recurrent_writer_node(input, value);
        rec_writers.push(Arc::new(MutCell::new(writer)));
    }
    rec_writers
}
