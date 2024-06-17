use std::{num::NonZeroUsize, sync::Arc};

use crate::{
    layers::residual::same_size_residual_layer,
    mut_cell::MutCell,
    node::SharedNode,
    nodes::{
        constant::constant_node,
        linear::{linear_layer, LinearLayerConfig},
        power::power_node,
        product::product_node,
        sin::sin_node,
        softmax::softmax_layer,
        sum::sum_node,
    },
    param::ParamInjection,
};

/// output shape: (word embedding depth, sequence length, heads)
pub fn transformer(
    inputs_seq: Vec<Vec<SharedNode>>,
    seq: SeqDef,
    depth: NonZeroUsize,
    heads: NonZeroUsize,
    mut param_injection: ParamInjection<'_>,
) -> Vec<Vec<Vec<SharedNode>>> {
    let word_embedding_seq = {
        let param_injection = param_injection.name_append(":word_embedding");
        linear_layer_seq(inputs_seq, depth, param_injection)
    };
    let layer_seq = positional_encoding(word_embedding_seq, seq);
    for layer in &layer_seq {
        assert_eq!(layer.len(), depth.get());
    }
    let mut self_attention_seqs = vec![];
    for i in 0..heads.get() {
        let param_injection = param_injection.name_append(&format!(":self_attention.{i}"));
        let seq = self_attention_seq(layer_seq.clone(), depth, param_injection);
        self_attention_seqs.push(seq);
    }
    let mut residual_connection_seqs = vec![];
    for self_attention_seq in self_attention_seqs {
        let mut residual_connection_seq = vec![];
        for (self_attention, x) in self_attention_seq.into_iter().zip(layer_seq.iter()) {
            let residual_connection = same_size_residual_layer(self_attention, x.clone());
            residual_connection_seq.push(residual_connection);
        }
        residual_connection_seqs.push(residual_connection_seq);
    }
    residual_connection_seqs
}

pub fn self_attention_seq(
    inputs_seq: Vec<Vec<SharedNode>>,
    depth: NonZeroUsize,
    mut param_injection: ParamInjection<'_>,
) -> Vec<Vec<SharedNode>> {
    let value_seq = {
        let param_injection = param_injection.name_append(":value");
        linear_layer_seq(inputs_seq.clone(), depth, param_injection)
    };
    let key_seq = {
        let param_injection = param_injection.name_append(":key");
        linear_layer_seq(inputs_seq.clone(), depth, param_injection)
    };
    let query_seq = {
        let param_injection = param_injection.name_append(":query");
        linear_layer_seq(inputs_seq.clone(), depth, param_injection)
    };

    let mut self_attention_seq = vec![];
    for query in query_seq {
        let mut similarity_scores = vec![];
        for key in &key_seq {
            let similarity_score = dot_product(&query, key);
            similarity_scores.push(similarity_score);
        }
        let similarity_prob = softmax_layer(similarity_scores);
        let mut scaled_value_seq = vec![];
        assert_eq!(value_seq.len(), similarity_prob.len());
        for (value, prob) in value_seq.iter().zip(similarity_prob.iter()) {
            let mut scaled_value = vec![];
            for value in value {
                scaled_value.push(Arc::new(MutCell::new(product_node(vec![
                    Arc::clone(value),
                    Arc::clone(prob),
                ]))));
            }
            scaled_value_seq.push(scaled_value);
        }
        let mut self_attention_value = vec![];
        for depth_pos in 0..depth.get() {
            let mut sum = vec![];
            for scaled_value in &scaled_value_seq {
                sum.push(Arc::clone(&scaled_value[depth_pos]));
            }
            let sum = Arc::new(MutCell::new(sum_node(sum)));
            self_attention_value.push(sum);
        }
        self_attention_seq.push(self_attention_value);
    }
    self_attention_seq
}

pub fn dot_product(a: &[SharedNode], b: &[SharedNode]) -> SharedNode {
    assert_eq!(a.len(), b.len());
    let mut products = vec![];
    for (a, b) in a.iter().zip(b.iter()) {
        let prod = Arc::new(MutCell::new(product_node(vec![
            Arc::clone(a),
            Arc::clone(b),
        ])));
        products.push(prod);
    }
    Arc::new(MutCell::new(sum_node(products)))
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

pub fn linear_layer_seq(
    inputs_seq: Vec<Vec<SharedNode>>,
    depth: NonZeroUsize,
    mut param_injection: ParamInjection<'_>,
) -> Vec<Vec<SharedNode>> {
    let mut outputs_seq = vec![];
    for inputs in inputs_seq {
        let config = LinearLayerConfig {
            depth,
            lambda: None,
        };
        let param_injection = param_injection.name_append(":seq_unit");
        let outputs = linear_layer(inputs, config, param_injection).unwrap();
        outputs_seq.push(outputs);
    }
    outputs_seq
}
