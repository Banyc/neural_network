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

/// gradient computation algorithm: BPTT
pub fn rnn(
    inputs_seq: Vec<Vec<SharedNode>>,
    depth: NonZeroUsize,
    activation: &Activation,
    mut param_injection: ParamInjection<'_>,
) -> Vec<Vec<SharedNode>> {
    assert!(!inputs_seq.is_empty());

    let mut rec_readers = vec![];
    let mut rec_values = vec![];
    for _ in 0..depth.get() {
        let (reader, value) = recurrent_reader_node();
        rec_readers.push(Arc::new(MutCell::new(reader)));
        rec_values.push(value);
    }

    let mut rnn_unit_seq = vec![rec_readers];

    let mut param_injection = param_injection.name_append(":rnn");
    for inputs in inputs_seq.into_iter() {
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
            linear_layer(
                rnn_unit_seq.last().unwrap().clone(),
                config,
                param_injection,
            )
            .unwrap()
        };
        let mut sum_layer = vec![];
        for (x, rec) in x.into_iter().zip(rec.into_iter()) {
            let node = sum_node(vec![x, rec]);
            sum_layer.push(Arc::new(MutCell::new(node)));
        }
        let act_layer = activation.activate(&sum_layer);
        assert_eq!(act_layer.len(), rec_values.len());
        rnn_unit_seq.push(act_layer);
    }

    let mut rec_writers = vec![];
    for (input, value) in rnn_unit_seq
        .pop()
        .unwrap()
        .into_iter()
        .zip(rec_values.into_iter())
    {
        let writer = recurrent_writer_node(input, value);
        rec_writers.push(Arc::new(MutCell::new(writer)));
    }
    rnn_unit_seq.push(rec_writers);

    rnn_unit_seq
}
