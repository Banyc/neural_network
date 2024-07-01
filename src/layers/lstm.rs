use graph::NodeIdx;

use crate::{
    layers::{activation::Activation, rnn::rnn_unit},
    node::GraphBuilder,
    nodes::{product::product_node, sum::sum_node},
    param::ParamInjection,
};

pub fn lstm(
    graph: &mut GraphBuilder,
    init_memory: LstmUnitMemory,
    inputs_seq: Vec<Vec<NodeIdx>>,
    mut param_injection: ParamInjection<'_>,
) -> Vec<Vec<NodeIdx>> {
    assert!(!inputs_seq.is_empty());

    let mut curr_memory = init_memory;
    let mut outputs_seq: Vec<Vec<NodeIdx>> = vec![];

    for inputs in inputs_seq.into_iter() {
        let param_injection = param_injection.name_append(":unit");
        let new_memory = lstm_unit(graph, inputs, curr_memory, param_injection);
        outputs_seq.push(new_memory.short_term_memory.clone());
        curr_memory = new_memory;
    }

    outputs_seq
}

#[derive(Debug)]
pub struct LstmUnitMemory {
    pub long_term_memory: Vec<NodeIdx>,
    pub short_term_memory: Vec<NodeIdx>,
}
fn lstm_unit(
    graph: &mut GraphBuilder,
    inputs: Vec<NodeIdx>,
    prev_memory: LstmUnitMemory,
    mut param_injection: ParamInjection<'_>,
) -> LstmUnitMemory {
    assert_eq!(
        prev_memory.long_term_memory.len(),
        prev_memory.short_term_memory.len()
    );

    // forget gate
    let long_term_remember_rate = {
        let param_injection = param_injection.name_append(":ltrr");
        let activation = Activation::Sigmoid;
        rnn_unit(
            graph,
            prev_memory.short_term_memory.clone(),
            inputs.clone(),
            &activation,
            param_injection,
        )
    };
    let mut remembered_long_term_memory = vec![];
    for (x, r) in prev_memory
        .long_term_memory
        .into_iter()
        .zip(long_term_remember_rate.into_iter())
    {
        let x = graph.insert_node(product_node(vec![x, r]));
        remembered_long_term_memory.push(x);
    }

    // input gate
    let potential_long_term_memory_to_remember_rate = {
        let param_injection = param_injection.name_append(":pltmtrr");
        let activation = Activation::Sigmoid;
        rnn_unit(
            graph,
            prev_memory.short_term_memory.clone(),
            inputs.clone(),
            &activation,
            param_injection,
        )
    };
    let potential_long_term_memory = {
        let param_injection = param_injection.name_append(":pltm");
        let activation = Activation::Tanh;
        rnn_unit(
            graph,
            prev_memory.short_term_memory.clone(),
            inputs.clone(),
            &activation,
            param_injection,
        )
    };
    let mut long_term_memory_adjustment = vec![];
    for (x, r) in potential_long_term_memory
        .into_iter()
        .zip(potential_long_term_memory_to_remember_rate.into_iter())
    {
        let x = graph.insert_node(product_node(vec![x, r]));
        long_term_memory_adjustment.push(x);
    }
    let mut new_long_term_memory = vec![];
    for (x, a) in remembered_long_term_memory
        .into_iter()
        .zip(long_term_memory_adjustment.into_iter())
    {
        let x = graph.insert_node(sum_node(vec![x, a]));
        new_long_term_memory.push(x);
    }

    // output gate
    let potential_short_term_memory_to_remember_rate = {
        let param_injection = param_injection.name_append(":pstmtrr");
        let activation = Activation::Sigmoid;
        rnn_unit(
            graph,
            prev_memory.short_term_memory,
            inputs,
            &activation,
            param_injection,
        )
    };
    let potential_short_term_memory = {
        let tanh = Activation::Tanh;
        graph.insert_nodes(tanh.activate(&new_long_term_memory))
    };
    let mut new_short_term_memory = vec![];
    for (x, r) in potential_short_term_memory
        .into_iter()
        .zip(potential_short_term_memory_to_remember_rate.into_iter())
    {
        let x = graph.insert_node(product_node(vec![x, r]));
        new_short_term_memory.push(x);
    }

    LstmUnitMemory {
        long_term_memory: new_long_term_memory,
        short_term_memory: new_short_term_memory,
    }
}
