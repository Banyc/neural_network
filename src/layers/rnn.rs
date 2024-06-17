use std::{num::NonZeroUsize, sync::Arc};

use crate::{
    layers::activation::Activation,
    mut_cell::MutCell,
    node::SharedNode,
    nodes::{
        linear::{linear_layer, LinearLayerConfig},
        sum::sum_node,
    },
    param::ParamInjection,
};

/// gradient computation algorithm: BPTT
pub fn rnn(
    init_hidden_states: Vec<SharedNode>,
    inputs_seq: Vec<Vec<SharedNode>>,
    activation: &Activation,
    mut param_injection: ParamInjection<'_>,
) -> Vec<Vec<SharedNode>> {
    assert!(!inputs_seq.is_empty());

    let mut init_hidden_states = Some(init_hidden_states);
    let mut rnn_unit_seq: Vec<Vec<SharedNode>> = vec![];

    for inputs in inputs_seq.into_iter() {
        let prev_hidden_states = match init_hidden_states.take() {
            Some(x) => x,
            None => rnn_unit_seq.last().unwrap().clone(),
        };
        let param_injection = param_injection.name_append(":unit");
        let depth = prev_hidden_states.len();
        let unit = rnn_unit(prev_hidden_states, inputs, activation, param_injection);
        assert_eq!(unit.len(), depth);
        rnn_unit_seq.push(unit);
    }

    rnn_unit_seq
}

fn rnn_unit(
    prev_hidden_states: Vec<SharedNode>,
    inputs: Vec<SharedNode>,
    activation: &Activation,
    mut param_injection: ParamInjection<'_>,
) -> Vec<SharedNode> {
    let depth = NonZeroUsize::new(prev_hidden_states.len()).unwrap();
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
        linear_layer(prev_hidden_states, config, param_injection).unwrap()
    };
    let mut sum_layer = vec![];
    for (x, rec) in x.into_iter().zip(rec.into_iter()) {
        let node = sum_node(vec![x, rec]);
        sum_layer.push(Arc::new(MutCell::new(node)));
    }
    let act_layer = activation.activate(&sum_layer);
    assert_eq!(act_layer.len(), depth.get());
    act_layer
}
