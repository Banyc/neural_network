use graph::NodeIdx;
use rand::Rng;
use thiserror::Error;
use vec_seg::SegKey;

use crate::{
    computation::{NodeBackpropagationComputation, NodeComputation, NodeScalarComputation},
    node::CompNode,
};

/// ```math
/// X \sim U(-\frac{1}{n^2}, \frac{1}{n^2})
/// ```
pub fn rnd_weights(op_len: usize) -> Vec<f64> {
    if op_len == 0 {
        return vec![];
    }
    let weight_bound = 1.0 / (op_len as f64).sqrt();
    let mut rng = rand::thread_rng();
    (0..op_len)
        .map(|_| {
            let weight: f64 = rng.gen_range(-weight_bound..weight_bound);
            weight
        })
        .collect()
}

/// ```math
/// f_w (x) = wx
/// ```
///
/// - `lambda`: for regularization
pub fn weight_node(
    operands: Vec<NodeIdx>,
    weights: SegKey,
    lambda: Option<f64>,
) -> Result<CompNode, WeightNodeError> {
    if operands.len() != weights.len() {
        return Err(WeightNodeError::ParameterSizeNotMatched);
    }
    let computation = WeightNodeComputation {
        lambda: lambda.unwrap_or(0.),
    };
    let node = CompNode::new(
        operands,
        NodeComputation::Scalar(Box::new(computation)),
        weights,
    );
    Ok(node)
}

#[derive(Debug, Clone)]
struct WeightNodeComputation {
    lambda: f64,
}
impl NodeScalarComputation for WeightNodeComputation {
    fn compute_output(
        &self,
        parameters: &[f64],
        operand_outputs: &[f64],
        _graph_inputs: &[f64],
    ) -> f64 {
        assert_eq!(operand_outputs.len(), parameters.len());
        weight(operand_outputs, parameters)
    }
}
impl NodeBackpropagationComputation for WeightNodeComputation {
    fn compute_gradient_of_this_at_operand(
        &self,
        parameters: &[f64],
        operand_outputs: &[f64],
        buf: Vec<f64>,
    ) -> Vec<f64> {
        assert_eq!(operand_outputs.len(), parameters.len());
        weight_derivative(parameters, buf)
    }

    fn compute_gradient_of_this_at_parameter(
        &self,
        parameters: &[f64],
        operand_outputs: &[f64],
        buf: Vec<f64>,
    ) -> Vec<f64> {
        assert_eq!(operand_outputs.len(), parameters.len());
        derivative_of_w(operand_outputs, buf)
    }

    fn regularization(&self, parameter: f64) -> f64 {
        derivative_of_l2_regularization(parameter, self.lambda)
    }
}

fn weight(x: &[f64], w: &[f64]) -> f64 {
    assert_eq!(x.len(), w.len());
    x.iter()
        .copied()
        .zip(w.iter().copied())
        .map(|(x, w)| x * w)
        .sum()
}

fn weight_derivative(w: &[f64], mut buf: Vec<f64>) -> Vec<f64> {
    buf.extend(w);
    buf
}

fn derivative_of_w(x: &[f64], mut buf: Vec<f64>) -> Vec<f64> {
    buf.extend(x);
    buf
}

fn derivative_of_l2_regularization(w: f64, lambda: f64) -> f64 {
    w * lambda
}

#[derive(Debug, Error)]
pub enum WeightNodeError {
    #[error("Parameter size not matched")]
    ParameterSizeNotMatched,
}

#[cfg(test)]
mod tests {
    use graph::dependency_order;

    use super::*;

    use crate::{
        computation::ComputationMode,
        node::{evaluate_once, GraphBuilder, NodeContext},
        nodes::input::{input_node_batch, InputNodeBatchParams},
        param::{ParamInjection, ParamInjector, Params},
    };

    fn assertion(assert_weight: impl Fn(&CompNode, &Params, &mut NodeContext)) {
        let mut params = ParamInjector::empty();
        let mut param_injection = ParamInjection {
            injector: &mut params,
            name: "".into(),
        };
        let mut graph = GraphBuilder::new();
        let input_nodes =
            graph.insert_nodes(input_node_batch(InputNodeBatchParams { start: 0, len: 3 }));
        let inputs = vec![1.0, 2.0, 3.0];
        let weights = param_injection
            .name_append(":weights")
            .get_or_create_params(|| [3.0, 2.0, 1.0].into_iter());
        let weight_node = graph.insert_node(weight_node(input_nodes, weights, None).unwrap());
        let mut graph = graph.build();
        let nodes_forward = dependency_order(&graph, &[weight_node]);
        let mut params = params.into_params();
        let mut cx = NodeContext::new();
        evaluate_once(
            &mut graph,
            &nodes_forward,
            &mut params,
            &[inputs],
            &mut cx,
            ComputationMode::Inference,
        );
        let weight_node = graph.nodes().get(weight_node).unwrap();
        assert_weight(weight_node, &params, &mut cx);
    }

    #[test]
    fn evaluate() {
        assertion(|weight_node, _, _| {
            let output = weight_node.output().unwrap()[0];
            assert_eq!(output, 3.0 * 1.0 + 2.0 * 2.0 + 1.0 * 3.0);
        });
    }

    #[test]
    fn gradient_of_this_at_operand() {
        assertion(|weight_node, params, cx| {
            let batch_index = 0;
            let ret = weight_node
                .gradient_of_this_at_operand(
                    batch_index,
                    params.seg().slice(weight_node.parameters()),
                    cx,
                )
                .unwrap();
            assert_eq!(&ret, &[3.0, 2.0, 1.0]);
        });
    }

    #[test]
    fn gradient_of_this_at_parameter() {
        assertion(|weight_node, params, cx| {
            let batch_index = 0;
            let ret = weight_node
                .gradient_of_this_at_parameter(
                    batch_index,
                    params.seg().slice(weight_node.parameters()),
                    cx,
                )
                .unwrap();
            assert_eq!(&ret, &[1.0, 2.0, 3.0]);
        });
    }
}
