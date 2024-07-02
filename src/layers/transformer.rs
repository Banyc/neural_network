use std::num::NonZeroUsize;

use graph::NodeIdx;

use crate::{
    layers::residual::same_size_residual_layer,
    node::GraphBuilder,
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
    graph: &mut GraphBuilder,
    attention_value: Vec<NodeIdx>,
    word_depth: NonZeroUsize,
    param_injection: ParamInjection<'_>,
) -> Vec<NodeIdx> {
    let config = LinearLayerConfig {
        depth: word_depth,
        lambda: None,
    };
    let scores = linear_layer(graph, attention_value, config, param_injection).unwrap();
    graph.insert_nodes(softmax_layer(scores))
}

#[derive(Debug)]
pub struct InputsSeq {
    pub inputs_seq: Vec<Vec<NodeIdx>>,
    pub seq: SeqDef,
}

/// output shape: (word length, sequence length)
pub fn codec_transformer(
    graph: &mut GraphBuilder,
    encoding: InputsSeq,
    decoding: InputsSeq,
    depth: NonZeroUsize,
    normalization: Normalization,
    mut param_injection: ParamInjection<'_>,
) -> Vec<Vec<NodeIdx>> {
    let mut decoding_word_len = None;
    for inputs in &decoding.inputs_seq {
        if let Some(len) = decoding_word_len {
            assert_eq!(len, inputs.len());
        }
        decoding_word_len = Some(inputs.len());
    }
    let decoding_word_len = NonZeroUsize::new(decoding_word_len.unwrap()).unwrap();

    let encoder_seq = {
        let param_injection = param_injection.name_append(":encoder");
        self_transformer(
            graph,
            encoding,
            depth,
            normalization.clone(),
            param_injection,
        )
    };
    let decoder_seq = {
        let param_injection = param_injection.name_append(":decoder");
        self_transformer(
            graph,
            decoding,
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
            residual_attention_seq(
                graph,
                reference,
                depth,
                normalization.clone(),
                param_injection,
            )
            .pop()
            .unwrap()
        };
        let word = {
            let param_injection = param_injection.name_append(":fc");
            attention_value_to_one_hot_word(
                graph,
                attention_value,
                decoding_word_len,
                param_injection,
            )
        };
        one_hot_words.push(word);
    }
    one_hot_words
}

/// output shape: (word embedding depth, sequence length)
pub fn self_transformer(
    graph: &mut GraphBuilder,
    inputs_seq: InputsSeq,
    depth: NonZeroUsize,
    normalization: Normalization,
    mut param_injection: ParamInjection<'_>,
) -> Vec<Vec<NodeIdx>> {
    let word_embedding_seq = {
        let param_injection = param_injection.name_append(":word_embedding");
        linear_layer_seq(graph, inputs_seq.inputs_seq, depth, param_injection)
    };
    let layer_seq = positional_encoding(graph, word_embedding_seq, inputs_seq.seq);
    for layer in &layer_seq {
        assert_eq!(layer.len(), depth.get());
    }
    let layer_seq = {
        let param_injection = param_injection.name_append(":norm");
        normalize_seq(graph, layer_seq, normalization.clone(), param_injection)
    };
    let reference = AttentionReference {
        referee_seq: layer_seq.clone(),
        referrer_seq: layer_seq.clone(),
    };
    {
        let param_injection = param_injection.name_append(":self_attention");
        residual_attention_seq(graph, reference, depth, normalization, param_injection)
    }
}

pub fn residual_attention_seq(
    graph: &mut GraphBuilder,
    reference: AttentionReference,
    depth: NonZeroUsize,
    normalization: Normalization,
    mut param_injection: ParamInjection<'_>,
) -> Vec<Vec<NodeIdx>> {
    let referrer_seq = reference.referrer_seq.clone();
    let attention_value_seq = {
        let param_injection = param_injection.name_append(":attention");
        attention_seq(graph, reference, depth, param_injection)
    };
    let attention_value_seq = {
        let param_injection = param_injection.name_append(":norm");
        normalize_seq(graph, attention_value_seq, normalization, param_injection)
    };
    let mut residual_connection_seq = vec![];
    for (attention_value, x) in attention_value_seq.into_iter().zip(referrer_seq.iter()) {
        let residual_connection =
            graph.insert_nodes(same_size_residual_layer(attention_value, x.clone()));
        residual_connection_seq.push(residual_connection);
    }
    residual_connection_seq
}

#[derive(Debug)]
pub struct AttentionReference {
    pub referee_seq: Vec<Vec<NodeIdx>>,
    pub referrer_seq: Vec<Vec<NodeIdx>>,
}
pub fn attention_seq(
    graph: &mut GraphBuilder,
    reference: AttentionReference,
    depth: NonZeroUsize,
    mut param_injection: ParamInjection<'_>,
) -> Vec<Vec<NodeIdx>> {
    let value_seq = {
        let param_injection = param_injection.name_append(":value");
        linear_layer_seq(graph, reference.referee_seq.clone(), depth, param_injection)
    };
    let key_seq = {
        let param_injection = param_injection.name_append(":key");
        linear_layer_seq(graph, reference.referee_seq.clone(), depth, param_injection)
    };
    let query_seq = {
        let param_injection = param_injection.name_append(":query");
        linear_layer_seq(graph, reference.referrer_seq, depth, param_injection)
    };

    let mut attention_seq = vec![];
    for query in query_seq {
        let mut similarity_scores = vec![];
        for key in &key_seq {
            let similarity_score = dot_product(graph, &query, key);
            similarity_scores.push(similarity_score);
        }
        let similarity_prob = graph.insert_nodes(softmax_layer(similarity_scores));
        let mut scaled_value_seq = vec![];
        assert_eq!(value_seq.len(), similarity_prob.len());
        for (value, &prob) in value_seq.iter().zip(similarity_prob.iter()) {
            let mut scaled_value = vec![];
            for &value in value {
                scaled_value.push(graph.insert_node(product_node(vec![value, prob])));
            }
            scaled_value_seq.push(scaled_value);
        }
        let mut self_attention_value = vec![];
        for depth_pos in 0..depth.get() {
            let mut sum = vec![];
            for scaled_value in &scaled_value_seq {
                sum.push(scaled_value[depth_pos]);
            }
            let sum = graph.insert_node(sum_node(sum));
            self_attention_value.push(sum);
        }
        attention_seq.push(self_attention_value);
    }
    attention_seq
}

pub fn dot_product(graph: &mut GraphBuilder, a: &[NodeIdx], b: &[NodeIdx]) -> NodeIdx {
    assert_eq!(a.len(), b.len());
    let mut products = vec![];
    for (a, b) in a.iter().copied().zip(b.iter().copied()) {
        let prod = graph.insert_node(product_node(vec![a, b]));
        products.push(prod);
    }
    graph.insert_node(sum_node(products))
}

#[derive(Debug, Clone)]
pub struct SeqDef {
    pub start_pos: NodeIdx,
    pub len: NodeIdx,
}
pub fn positional_encoding(
    graph: &mut GraphBuilder,
    inputs_seq: Vec<Vec<NodeIdx>>,
    seq: SeqDef,
) -> Vec<Vec<NodeIdx>> {
    let mut outputs_seq = vec![];
    for (seq_off, inputs) in inputs_seq.into_iter().enumerate() {
        let mut unit = vec![];
        let depth_len = NonZeroUsize::new(inputs.len()).unwrap();
        for (depth_pos, node) in inputs.into_iter().enumerate() {
            let seq_off = graph.insert_node(constant_node(seq_off as f64));
            let seq_pos = graph.insert_node(sum_node(vec![seq.start_pos, seq_off]));
            let pos = SeqDepthPosition {
                seq_pos,
                seq_len: seq.len,
                depth_pos,
                depth_len,
            };
            let pos = embedding_position(graph, pos);
            let node = graph.insert_node(sum_node(vec![node, pos]));
            unit.push(node);
        }
        outputs_seq.push(unit);
    }
    outputs_seq
}

#[derive(Debug, Clone)]
pub struct SeqDepthPosition {
    pub seq_pos: NodeIdx,
    pub seq_len: NodeIdx,
    pub depth_pos: usize,
    pub depth_len: NonZeroUsize,
}
pub fn embedding_position(graph: &mut GraphBuilder, pos: SeqDepthPosition) -> NodeIdx {
    let pow = -((2 * pos.depth_pos) as f64 / pos.depth_len.get() as f64);
    let pow = graph.insert_node(power_node(pos.seq_len, pow));
    let prod = graph.insert_node(product_node(vec![pos.seq_pos, pow]));
    graph.insert_node(sin_node(prod))
}

pub fn linear_layer_seq(
    graph: &mut GraphBuilder,
    inputs_seq: Vec<Vec<NodeIdx>>,
    depth: NonZeroUsize,
    mut param_injection: ParamInjection<'_>,
) -> Vec<Vec<NodeIdx>> {
    let mut outputs_seq = vec![];
    for inputs in inputs_seq {
        let config = LinearLayerConfig {
            depth,
            lambda: None,
        };
        let param_injection = param_injection.name_append(":seq_unit");
        let outputs = linear_layer(graph, inputs, config, param_injection).unwrap();
        outputs_seq.push(outputs);
    }
    outputs_seq
}

pub fn normalize_seq(
    graph: &mut GraphBuilder,
    inputs_seq: Vec<Vec<NodeIdx>>,
    normalization: Normalization,
    mut param_injection: ParamInjection<'_>,
) -> Vec<Vec<NodeIdx>> {
    let mut normalized_seq = vec![];
    for layer in inputs_seq {
        let param_injection = param_injection.name_append(":norm_unit");
        let layer = normalization
            .clone()
            .normalize(graph, layer, param_injection);
        let layer = graph.insert_nodes(layer);
        normalized_seq.push(layer);
    }
    normalized_seq
}

#[cfg(test)]
mod tests {
    use crate::{
        neural_network::{AccurateFnParams, NeuralNetwork, TrainOption},
        nodes::{input::InputNodeGen, log_loss::log_loss_node},
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

        let mut graph = GraphBuilder::new();
        let hello = [1, 0, 0].iter().map(|&x| x as f64);
        let world = [0, 1, 0].iter().map(|&x| x as f64);
        let eos = [0, 0, 1].iter().map(|&x| x as f64);

        let mut input_gen = InputNodeGen::new();
        // hello, world, EOS
        let encoding_inputs_seq = vec![
            graph.insert_nodes(input_gen.gen(3)),
            graph.insert_nodes(input_gen.gen(3)),
            graph.insert_nodes(input_gen.gen(3)),
        ];
        let encoding_seq = SeqDef {
            start_pos: graph.insert_node(constant_node(0.)),
            len: graph.insert_node(constant_node(3.)),
        };
        let encoding = InputsSeq {
            inputs_seq: encoding_inputs_seq,
            seq: encoding_seq,
        };
        // EOS, hello, world
        let decoding_inputs_seq = vec![
            graph.insert_nodes(input_gen.gen(3)),
            graph.insert_nodes(input_gen.gen(3)),
            graph.insert_nodes(input_gen.gen(3)),
        ];
        let decoding_seq = SeqDef {
            start_pos: graph.insert_node(constant_node(0.)),
            len: graph.insert_node(constant_node(3.)),
        };
        let decoding = InputsSeq {
            inputs_seq: decoding_inputs_seq,
            seq: decoding_seq,
        };
        let depth = NonZeroUsize::new(5).unwrap();
        let normalization = Normalization::LayerNorm;

        let output_word_seq = codec_transformer(
            &mut graph,
            encoding,
            decoding,
            depth,
            normalization,
            param_injection,
        );
        dbg!(&output_word_seq);
        let terminal_nodes = output_word_seq
            .iter()
            .flat_map(|x| x.iter())
            .copied()
            .collect::<Vec<NodeIdx>>();

        // hello, world, EOS
        let label_seq = vec![
            graph.insert_nodes(input_gen.gen(3)),
            graph.insert_nodes(input_gen.gen(3)),
            graph.insert_nodes(input_gen.gen(3)),
        ];

        let mut error_inputs = vec![];
        error_inputs.extend(label_seq.into_iter().flat_map(|x| x.into_iter()));
        error_inputs.extend(terminal_nodes.clone());
        let error_node = graph.insert_node(log_loss_node(error_inputs));
        let graph = graph.build();

        let mut nn = NeuralNetwork::new(graph, terminal_nodes, error_node);

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
