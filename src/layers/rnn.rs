use std::num::NonZeroUsize;

use graph::NodeIdx;

use crate::{
    layers::activation::Activation,
    node::GraphBuilder,
    nodes::{
        linear::{linear_layer, LinearLayerConfig},
        sum::sum_node,
    },
    param::ParamInjection,
};

/// gradient computation algorithm: BPTT
pub fn rnn(
    graph: &mut GraphBuilder,
    init_hidden_states: Vec<NodeIdx>,
    inputs_seq: Vec<Vec<NodeIdx>>,
    activation: &Activation,
    mut param_injection: ParamInjection<'_>,
) -> Vec<Vec<NodeIdx>> {
    assert!(!inputs_seq.is_empty());

    let mut hidden_states = init_hidden_states;
    let mut outputs_seq: Vec<Vec<NodeIdx>> = vec![];

    for inputs in inputs_seq.into_iter() {
        let param_injection = param_injection.name_append(":unit");
        let depth = hidden_states.len();
        let outputs = rnn_unit(graph, hidden_states, inputs, activation, param_injection);
        hidden_states = outputs.clone();
        assert_eq!(outputs.len(), depth);
        outputs_seq.push(outputs);
    }

    outputs_seq
}

pub fn rnn_unit(
    graph: &mut GraphBuilder,
    prev_hidden_states: Vec<NodeIdx>,
    inputs: Vec<NodeIdx>,
    activation: &Activation,
    mut param_injection: ParamInjection<'_>,
) -> Vec<NodeIdx> {
    let depth = NonZeroUsize::new(prev_hidden_states.len()).unwrap();
    let x = {
        let param_injection = param_injection.name_append(":input");
        let config = LinearLayerConfig {
            depth,
            lambda: None,
        };
        linear_layer(graph, inputs, config, param_injection).unwrap()
    };
    let rec = {
        let param_injection = param_injection.name_append(":rec");
        let config = LinearLayerConfig {
            depth,
            lambda: None,
        };
        linear_layer(graph, prev_hidden_states, config, param_injection).unwrap()
    };
    let mut sum_layer = vec![];
    for (x, rec) in x.into_iter().zip(rec.into_iter()) {
        let node = graph.insert_node(sum_node(vec![x, rec]));
        sum_layer.push(node);
    }
    let act_layer = graph.insert_nodes(activation.activate(&sum_layer));
    assert_eq!(act_layer.len(), depth.get());
    act_layer
}
