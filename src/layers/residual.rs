use std::num::NonZeroUsize;

use crate::{
    mut_cell::MutCell,
    node::SharedNode,
    nodes::{
        linear::{linear_layer, LinearLayerConfig},
        sum::sum_node,
    },
    param::ParamInjection,
    ref_ctr::RefCtr,
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
    param_injection: ParamInjection<'_>,
) -> Vec<SharedNode> {
    assert!(!outputs.is_empty());
    assert!(!inputs.is_empty());
    let inputs = if outputs.len() != inputs.len() {
        let config = LinearLayerConfig {
            depth: NonZeroUsize::new(outputs.len()).unwrap(),
            lambda: None,
        };
        linear_layer(inputs, config, param_injection).unwrap()
    } else {
        inputs
    };
    same_size_residual_layer(outputs, inputs)
}

/// ```math
/// f(o, i) = o + i
/// ```
pub fn same_size_residual_layer(
    outputs: Vec<SharedNode>,
    inputs: Vec<SharedNode>,
) -> Vec<SharedNode> {
    assert!(!outputs.is_empty());
    assert!(!inputs.is_empty());
    assert_eq!(outputs.len(), inputs.len());
    let mut layer = vec![];
    for (o, i) in outputs.into_iter().zip(inputs.into_iter()) {
        let node = sum_node(vec![o, i]);
        layer.push(RefCtr::new(MutCell::new(node)));
    }
    layer
}
