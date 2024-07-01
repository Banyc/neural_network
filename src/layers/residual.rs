use std::num::NonZeroUsize;

use graph::NodeIdx;

use crate::{
    node::{CompNode, GraphBuilder},
    nodes::{
        linear::{linear_layer, LinearLayerConfig},
        sum::sum_node,
    },
    param::ParamInjection,
};

/// ```math
/// f(o, i) = \begin{cases}
///   o + i    & \|o\| = \|i\| \\
///   o + l(i) & \|o\| \neq \|i\| \\
/// \end{cases}
/// ```
///
/// - $l$: linear layer
pub fn residual_layer(
    graph: &mut GraphBuilder,
    outputs: Vec<NodeIdx>,
    inputs: Vec<NodeIdx>,
    param_injection: ParamInjection<'_>,
) -> Vec<NodeIdx> {
    assert!(!outputs.is_empty());
    assert!(!inputs.is_empty());
    let inputs = if outputs.len() != inputs.len() {
        let config = LinearLayerConfig {
            depth: NonZeroUsize::new(outputs.len()).unwrap(),
            lambda: None,
        };
        linear_layer(graph, inputs, config, param_injection).unwrap()
    } else {
        inputs
    };
    graph.insert_nodes(same_size_residual_layer(outputs, inputs))
}

/// ```math
/// f(o, i) = o + i
/// ```
pub fn same_size_residual_layer(outputs: Vec<NodeIdx>, inputs: Vec<NodeIdx>) -> Vec<CompNode> {
    assert!(!outputs.is_empty());
    assert!(!inputs.is_empty());
    assert_eq!(outputs.len(), inputs.len());
    let mut layer = vec![];
    for (o, i) in outputs.into_iter().zip(inputs.into_iter()) {
        let node = sum_node(vec![o, i]);
        layer.push(node);
    }
    layer
}
