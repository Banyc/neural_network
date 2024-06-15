use std::{num::NonZeroUsize, sync::Arc};

use crate::{
    mut_cell::MutCell,
    node::SharedNode,
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
    outputs: Vec<SharedNode>,
    inputs: Vec<SharedNode>,
    param_injection: Option<ParamInjection<'_>>,
) -> Vec<SharedNode> {
    assert!(!outputs.is_empty());
    assert!(!inputs.is_empty());
    let inputs = if outputs.len() != inputs.len() {
        let config = LinearLayerConfig {
            depth: NonZeroUsize::new(outputs.len()).unwrap(),
            initial_weights: None,
            initial_bias: None,
            lambda: None,
        };
        linear_layer(inputs, config, param_injection).unwrap()
    } else {
        inputs
    };
    assert_eq!(outputs.len(), inputs.len());
    let mut layer = vec![];
    for (o, i) in outputs.into_iter().zip(inputs.into_iter()) {
        let node = sum_node(vec![o, i]);
        layer.push(Arc::new(MutCell::new(node)));
    }
    layer
}
