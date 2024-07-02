use graph::NodeIdx;
use vec_seg::SegKey;

use crate::{
    computation::{
        ComputationMode, NodeBackpropagationComputation, NodeBatchComputation, NodeComputation,
    },
    mut_cell::MutCell,
    node::CompNode,
    param::{ParamInjection, Params},
    ref_ctr::RefCtr,
    tensor::Shape,
};

pub fn default_trainable_params() -> Vec<f64> {
    let beta = 0.;
    let gamma = 1.;
    let params = vec![beta, gamma];
    params
}
pub fn default_saved_params() -> Vec<f64> {
    let mean = 0.;
    let std_dev = 1.;
    let params = vec![mean, std_dev];
    params
}

/// ```math
/// f_{\beta, \gamma} (x) = \frac{x - \mu_x}{\sigma_x} \gamma + \beta
/// ```
pub fn batch_norm_node(
    operand: NodeIdx,
    saved_params: SegKey,
    trainable_params: SegKey,
    alpha: f64,
) -> CompNode {
    let computation = BatchNormComputation {
        saved_params,
        alpha,
    };
    CompNode::new(
        vec![operand],
        RefCtr::new(MutCell::new(NodeComputation::Batch(Box::new(computation)))),
        trainable_params,
    )
}

#[derive(Clone)]
pub struct BatchNormLayerConfig {
    pub alpha: f64,
}
pub fn batch_norm_layer(
    input_nodes: Vec<NodeIdx>,
    config: BatchNormLayerConfig,
    mut param_injection: ParamInjection<'_>,
) -> Vec<CompNode> {
    let mut layer = Vec::with_capacity(input_nodes.len());
    for (i, input_node) in input_nodes.into_iter().enumerate() {
        let mut param_injection = param_injection.name_append(&format!(":bn.{i}"));
        let saved_params = param_injection
            .name_append(":saved")
            .get_or_create_params(|| default_saved_params().into_iter());
        let trainable_params = param_injection
            .name_append(":trainable")
            .get_or_create_params(|| default_trainable_params().into_iter());
        let batch_norm_node =
            batch_norm_node(input_node, saved_params, trainable_params, config.alpha);
        layer.push(batch_norm_node);
    }
    layer
}

#[derive(Debug)]
struct BatchNormComputation {
    /// $(mean, std)$
    saved_params: SegKey,
    alpha: f64,
}
impl NodeBatchComputation for BatchNormComputation {
    fn compute_output(
        &mut self,
        params: &mut Params,
        param_key: SegKey,
        operand_outputs: &[f64],
        operand_outputs_shape: &Shape,
        mut buf: Vec<f64>,
        mode: ComputationMode,
    ) -> Vec<f64> {
        assert_eq!(param_key.len(), 2);
        let parameters = params.seg().slice(param_key);
        let beta = parameters[0];
        let gamma = parameters[1];

        let operand_len = operand_outputs_shape[0];
        assert_eq!(operand_len, 1);
        let batch_size = operand_outputs_shape[1];
        assert_eq!(operand_outputs.len(), operand_len * batch_size);
        let each_operand = (0..batch_size)
            .map(|batch_index| batch_index * batch_size)
            .map(|i| operand_outputs[i]);

        match mode {
            ComputationMode::Training => {
                let mean = each_operand
                    .clone()
                    .map(|x| x / batch_size as f64)
                    .sum::<f64>();
                let var = each_operand
                    .clone()
                    .map(|x| x - mean)
                    .map(|x| x.powi(2))
                    .map(|x| x / batch_size as f64)
                    .sum::<f64>();
                let std_dev = var.sqrt();

                let normalized = each_operand.map(|x| x - mean).map(|x| x / std_dev);
                let scaled_shifted = normalized.map(|x| x * gamma + beta);

                let saved_params = params.seg_mut().slice_mut(self.saved_params);
                let ema_mean = &mut saved_params[0];
                *ema_mean = *ema_mean * self.alpha + mean * (1. - self.alpha);
                let ema_std_dev = &mut saved_params[1];
                *ema_std_dev = *ema_std_dev * self.alpha + std_dev * (1. - self.alpha);

                buf.extend(scaled_shifted);
                buf
            }
            ComputationMode::Inference => {
                let saved_params = params.seg().slice(self.saved_params);
                let mean = saved_params[0];
                let std_dev = saved_params[1];

                let normalized = each_operand.map(|x| x - mean).map(|x| x / std_dev);
                let scaled_shifted = normalized.map(|x| x * gamma + beta);

                buf.extend(scaled_shifted);
                buf
            }
        }
    }
}
impl NodeBackpropagationComputation for BatchNormComputation {
    fn compute_gradient_of_this_at_operand(
        &self,
        parameters: &[f64],
        operand_outputs: &[f64],
        mut buf: Vec<f64>,
    ) -> Vec<f64> {
        assert_eq!(parameters.len(), 2);
        let _beta = parameters[0];
        let gamma = parameters[1];
        assert_eq!(operand_outputs.len(), 1);
        buf.extend([gamma]);
        buf
    }

    fn compute_gradient_of_this_at_parameter(
        &self,
        parameters: &[f64],
        operand_outputs: &[f64],
        mut buf: Vec<f64>,
    ) -> Vec<f64> {
        assert_eq!(parameters.len(), 2);
        let _beta = parameters[0];
        let _gamma = parameters[1];
        assert_eq!(operand_outputs.len(), 1);
        buf.extend([operand_outputs[0], 1.]);
        buf
    }
}
