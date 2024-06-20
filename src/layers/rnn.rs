use std::num::NonZeroUsize;

use crate::{
    layers::activation::Activation,
    mut_cell::MutCell,
    node::SharedNode,
    nodes::{
        linear::{linear_layer, LinearLayerConfig},
        sum::sum_node,
    },
    param::ParamInjection,
    ref_ctr::RefCtr,
};

/// gradient computation algorithm: BPTT
pub fn rnn(
    init_hidden_states: Vec<SharedNode>,
    inputs_seq: Vec<Vec<SharedNode>>,
    activation: &Activation,
    mut param_injection: ParamInjection<'_>,
) -> Vec<Vec<SharedNode>> {
    assert!(!inputs_seq.is_empty());

    let mut hidden_states = init_hidden_states;
    let mut outputs_seq: Vec<Vec<SharedNode>> = vec![];

    for inputs in inputs_seq.into_iter() {
        let param_injection = param_injection.name_append(":unit");
        let depth = hidden_states.len();
        let outputs = rnn_unit(hidden_states, inputs, activation, param_injection);
        hidden_states = outputs.clone();
        assert_eq!(outputs.len(), depth);
        outputs_seq.push(outputs);
    }

    outputs_seq
}

pub fn rnn_unit(
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
        sum_layer.push(RefCtr::new(MutCell::new(node)));
    }
    let act_layer = activation.activate(&sum_layer);
    assert_eq!(act_layer.len(), depth.get());
    act_layer
}
