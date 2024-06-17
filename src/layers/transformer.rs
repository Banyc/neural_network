use std::{num::NonZeroUsize, sync::Arc};

use crate::{
    mut_cell::MutCell,
    node::SharedNode,
    nodes::{
        constant::constant_node,
        linear::{linear_layer, LinearLayerConfig},
        power::power_node,
        product::product_node,
        sin::sin_node,
        sum::sum_node,
    },
    param::ParamInjection,
};

pub fn transformer(
    inputs_seq: Vec<Vec<SharedNode>>,
    seq: SeqDef,
    depth: NonZeroUsize,
    mut param_injection: ParamInjection<'_>,
) {
    let layer = {
        let param_injection = param_injection.name_append(":we");
        word_embedding(inputs_seq, depth, param_injection)
    };
    let layer = positional_encoding(layer, seq);
    todo!()
}

pub struct SeqDef {
    pub start_pos: SharedNode,
    pub len: SharedNode,
}
pub fn positional_encoding(inputs_seq: Vec<Vec<SharedNode>>, seq: SeqDef) -> Vec<Vec<SharedNode>> {
    let mut outputs_seq = vec![];
    for (seq_off, inputs) in inputs_seq.into_iter().enumerate() {
        let mut unit = vec![];
        let depth_len = NonZeroUsize::new(inputs.len()).unwrap();
        for (depth_pos, node) in inputs.into_iter().enumerate() {
            let seq_off = Arc::new(MutCell::new(constant_node(seq_off as f64)));
            let seq_pos = Arc::new(MutCell::new(sum_node(vec![seq.start_pos.clone(), seq_off])));
            let pos = SeqDepthPosition {
                seq_pos,
                seq_len: seq.len.clone(),
                depth_pos,
                depth_len,
            };
            let pos = embedding_position(pos);
            let node = Arc::new(MutCell::new(sum_node(vec![node, pos])));
            unit.push(node);
        }
        outputs_seq.push(unit);
    }
    outputs_seq
}

#[derive(Debug, Clone)]
pub struct SeqDepthPosition {
    pub seq_pos: SharedNode,
    pub seq_len: SharedNode,
    pub depth_pos: usize,
    pub depth_len: NonZeroUsize,
}
pub fn embedding_position(pos: SeqDepthPosition) -> SharedNode {
    let pow = -((2 * pos.depth_pos) as f64 / pos.depth_len.get() as f64);
    let pow = Arc::new(MutCell::new(power_node(pos.seq_len, pow)));
    let prod = Arc::new(MutCell::new(product_node(vec![pos.seq_pos, pow])));
    Arc::new(MutCell::new(sin_node(prod)))
}

pub fn word_embedding(
    inputs_seq: Vec<Vec<SharedNode>>,
    depth: NonZeroUsize,
    mut param_injection: ParamInjection<'_>,
) -> Vec<Vec<SharedNode>> {
    let mut outputs_seq = vec![];
    for inputs in inputs_seq {
        let param_injection = param_injection.name_append(":token");
        let config = LinearLayerConfig {
            depth,
            lambda: None,
        };
        let outputs = linear_layer(inputs, config, param_injection).unwrap();
        outputs_seq.push(outputs);
    }
    outputs_seq
}
