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

use super::norm::Normalization;

pub fn attention_value_to_one_hot_word(
    attention_value: Vec<SharedNode>,
    word_depth: NonZeroUsize,
    param_injection: ParamInjection<'_>,
) -> Vec<SharedNode> {
    let config = LinearLayerConfig {
        depth: word_depth,
        lambda: None,
    };
    let scores = linear_layer(attention_value, config, param_injection).unwrap();
    softmax_layer(scores)
}

/// output shape: (word length, sequence length)
pub fn codec_transformer(
    encoding_inputs_seq: Vec<Vec<SharedNode>>,
    encoding_seq: SeqDef,
    decoding_inputs_seq: Vec<Vec<SharedNode>>,
    decoding_seq: SeqDef,
    depth: NonZeroUsize,
    normalization: Normalization,
    mut param_injection: ParamInjection<'_>,
) -> Vec<Vec<SharedNode>> {
    let mut decoding_word_len = None;
    for inputs in &decoding_inputs_seq {
        if let Some(len) = decoding_word_len {
            assert_eq!(len, inputs.len());
        }
        decoding_word_len = Some(inputs.len());
    }
    let decoding_word_len = NonZeroUsize::new(decoding_word_len.unwrap()).unwrap();

    let encoder_seq = {
        let param_injection = param_injection.name_append(":encoder");
        self_transformer(
            encoding_inputs_seq,
            encoding_seq,
            depth,
            normalization.clone(),
            param_injection,
        )
    };
    let decoder_seq = {
        let param_injection = param_injection.name_append(":decoder");
        self_transformer(
            decoding_inputs_seq,
            decoding_seq,
            depth,
            normalization.clone(),
            param_injection,
        )
    };
    let mut one_hot_words = vec![];
    for decoder in decoder_seq {
        let reference = AttentionReference {
            referee_seq: encoder_seq.clone(),
            referrer_seq: vec![decoder],
        };
        let attention_value = {
            let param_injection = param_injection.name_append(":codec_attention");
            residual_attention_seq(reference, depth, normalization.clone(), param_injection)
                .pop()
                .unwrap()
        };
        let word = {
            let param_injection = param_injection.name_append(":fc");
            attention_value_to_one_hot_word(attention_value, decoding_word_len, param_injection)
        };
        one_hot_words.push(word);
    }
    one_hot_words
}

/// output shape: (word embedding depth, sequence length)
pub fn self_transformer(
    inputs_seq: Vec<Vec<SharedNode>>,
    seq: SeqDef,
    depth: NonZeroUsize,
    normalization: Normalization,
    mut param_injection: ParamInjection<'_>,
) -> Vec<Vec<SharedNode>> {
    let word_embedding_seq = {
        let param_injection = param_injection.name_append(":word_embedding");
        linear_layer_seq(inputs_seq, depth, param_injection)
    };
    let layer_seq = positional_encoding(word_embedding_seq, seq);
    for layer in &layer_seq {
        assert_eq!(layer.len(), depth.get());
    }
    let layer_seq = {
        let param_injection = param_injection.name_append(":norm");
        normalize_seq(layer_seq, normalization.clone(), param_injection)
    };
    let reference = AttentionReference {
        referee_seq: layer_seq.clone(),
        referrer_seq: layer_seq.clone(),
    };
    {
        let param_injection = param_injection.name_append(":self_attention");
        residual_attention_seq(reference, depth, normalization, param_injection)
    }
}

pub fn residual_attention_seq(
    reference: AttentionReference,
    depth: NonZeroUsize,
    normalization: Normalization,
    mut param_injection: ParamInjection<'_>,
) -> Vec<Vec<SharedNode>> {
    let referrer_seq = reference.referrer_seq.clone();
    let attention_value_seq = {
        let param_injection = param_injection.name_append(":attention");
        attention_seq(reference, depth, param_injection)
    };
    let attention_value_seq = {
        let param_injection = param_injection.name_append(":norm");
        normalize_seq(attention_value_seq, normalization, param_injection)
    };
    let mut residual_connection_seq = vec![];
    for (attention_value, x) in attention_value_seq.into_iter().zip(referrer_seq.iter()) {
        let residual_connection = same_size_residual_layer(attention_value, x.clone());
        residual_connection_seq.push(residual_connection);
    }
    residual_connection_seq
}

#[derive(Debug)]
pub struct AttentionReference {
    pub referee_seq: Vec<Vec<SharedNode>>,
    pub referrer_seq: Vec<Vec<SharedNode>>,
}
pub fn attention_seq(
    reference: AttentionReference,
    depth: NonZeroUsize,
    mut param_injection: ParamInjection<'_>,
) -> Vec<Vec<SharedNode>> {
    let value_seq = {
        let param_injection = param_injection.name_append(":value");
        linear_layer_seq(reference.referee_seq.clone(), depth, param_injection)
    };
    let key_seq = {
        let param_injection = param_injection.name_append(":key");
        linear_layer_seq(reference.referee_seq.clone(), depth, param_injection)
    };
    let query_seq = {
        let param_injection = param_injection.name_append(":query");
        linear_layer_seq(reference.referrer_seq, depth, param_injection)
    };

    let mut attention_seq = vec![];
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
        attention_seq.push(self_attention_value);
    }
    attention_seq
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

#[derive(Debug, Clone)]
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

pub fn normalize_seq(
    inputs_seq: Vec<Vec<SharedNode>>,
    normalization: Normalization,
    mut param_injection: ParamInjection<'_>,
) -> Vec<Vec<SharedNode>> {
    let mut normalized_seq = vec![];
    for layer in inputs_seq {
        let param_injection = param_injection.name_append(":norm_unit");
        let layer = normalization.clone().normalize(layer, param_injection);
        normalized_seq.push(layer);
    }
    normalized_seq
}

#[cfg(test)]
mod tests {
    use crate::{
        neural_network::{AccurateFnParams, NeuralNetwork, TrainOption},
        nodes::{input::InputNodeGen, mse::mse_node},
        param::ParamInjector,
        tests::max_i,
    };

    use super::*;

    #[test]
    fn test_converge() {
        let mut param_injector = ParamInjector::empty();
        let param_injection = ParamInjection {
            injector: &mut param_injector,
            name: "".to_string(),
        };

        let hello = [1, 0, 0].iter().map(|&x| x as f64);
        let world = [0, 1, 0].iter().map(|&x| x as f64);
        let eos = [0, 0, 1].iter().map(|&x| x as f64);

        let mut input_gen = InputNodeGen::new();
        // hello, world, EOS
        let encoding_inputs_seq = vec![input_gen.gen(3), input_gen.gen(3), input_gen.gen(3)];
        let encoding_seq = SeqDef {
            start_pos: Arc::new(MutCell::new(constant_node(0.))),
            len: Arc::new(MutCell::new(constant_node(3.))),
        };
        // EOS, hello, world
        let decoding_inputs_seq = vec![input_gen.gen(3), input_gen.gen(3), input_gen.gen(3)];
        let decoding_seq = SeqDef {
            start_pos: Arc::new(MutCell::new(constant_node(0.))),
            len: Arc::new(MutCell::new(constant_node(3.))),
        };
        let depth = NonZeroUsize::new(5).unwrap();
        let normalization = Normalization::LayerNorm;

        let output_word_seq = codec_transformer(
            encoding_inputs_seq,
            encoding_seq,
            decoding_inputs_seq,
            decoding_seq,
            depth,
            normalization,
            param_injection,
        );
        dbg!(&output_word_seq);
        let terminal_nodes = output_word_seq
            .iter()
            .flat_map(|x| x.iter())
            .map(Arc::clone)
            .collect::<Vec<SharedNode>>();

        // hello, world, EOS
        let label_seq = vec![input_gen.gen(3), input_gen.gen(3), input_gen.gen(3)];

        let mut error_inputs = vec![];
        error_inputs.extend(label_seq.into_iter().flat_map(|x| x.into_iter()));
        error_inputs.extend(terminal_nodes.clone());
        let error_node = Arc::new(MutCell::new(mse_node(error_inputs)));

        let mut nn = NeuralNetwork::new(terminal_nodes, error_node);

        let mut dataset = vec![];

        {
            let mut sample = vec![];

            // encoding
            sample.extend(hello.clone());
            sample.extend(world.clone());
            sample.extend(eos.clone());
            // decoding
            sample.extend(eos.clone());
            sample.extend(hello.clone());
            sample.extend(world.clone());
            // label
            sample.extend(hello.clone());
            sample.extend(world.clone());
            sample.extend(eos.clone());

            dataset.push(sample);
        }
        {
            let mut sample = vec![];

            // encoding
            sample.extend(world.clone());
            sample.extend(hello.clone());
            sample.extend(eos.clone());
            // decoding
            sample.extend(eos.clone());
            sample.extend(world.clone());
            sample.extend(hello.clone());
            // label
            sample.extend(world.clone());
            sample.extend(hello.clone());
            sample.extend(eos.clone());

            dataset.push(sample);
        }

        let eval = nn.evaluate(&dataset[0..1]);
        println!("{eval:?}");

        let option = TrainOption::StochasticGradientDescent;
        nn.train(&dataset, 0.1, 1024 * 2, option);

        let eval = nn.evaluate(&dataset[0..1]);
        println!("{eval:?}");
        let eval = nn.evaluate(&dataset[1..2]);
        println!("{eval:?}");

        let acc_config = SeqAcc {
            word_len: 3,
            words: 3,
        };
        let acc = nn.accuracy(&dataset[0..1], |x| accurate(x, &acc_config));
        assert_eq!(acc, 1.);
    }

    #[derive(Debug, Clone)]
    pub struct SeqAcc {
        pub word_len: usize,
        pub words: usize,
    }
    pub fn accurate(params: AccurateFnParams, seq_config: &SeqAcc) -> bool {
        let eval = params.outputs.chunks(seq_config.word_len);
        let label = &params.inputs[params.inputs.len() - seq_config.word_len * seq_config.words..];
        let label = label.chunks(seq_config.word_len);
        for (eval, label) in eval.zip(label) {
            assert_eq!(eval.len(), label.len());
            let is_acc = max_i(eval) == max_i(label);
            if !is_acc {
                return false;
            }
        }
        true
    }
}
