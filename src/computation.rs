use primitive::vec_seg::SegKey;

use crate::{param::Params, tensor::Shape};

/// The function of this node should be
/// ```math
/// f : \mathbb{R}^n \to \mathbb{R}
/// ```
pub trait NodeScalarComputation: core::fmt::Debug + NodeBackpropagationComputation {
    fn compute_output(
        &self,
        parameters: &[f64],
        operand_outputs: &[f64],
        graph_inputs: &[f64],
    ) -> f64;
}

pub trait NodeBatchComputation: core::fmt::Debug + NodeBackpropagationComputation {
    fn compute_output(
        &mut self,
        params: &mut Params,
        param_key: SegKey,
        operand_outputs: &[f64],
        operand_outputs_shape: &Shape,
        buf: Vec<f64>,
        mode: ComputationMode,
    ) -> Vec<f64>;
}

pub trait NodeBackpropagationComputation: core::fmt::Debug + Sync + Send + 'static {
    /// ```math
    /// \frac{\partial f}{\partial z}
    /// ```
    ///
    /// - $z$: the non-tunable operands of this node
    /// - $f$: this node
    fn compute_gradient_of_this_at_operand(
        &self,
        parameters: &[f64],
        operand_outputs: &[f64],
        buf: Vec<f64>,
    ) -> Vec<f64>;

    /// ```math
    /// \frac{\partial f}{\partial w}
    /// ```
    ///
    /// - $w$: the tunable parameters of this node
    /// - $f$: this node
    fn compute_gradient_of_this_at_parameter(
        &self,
        parameters: &[f64],
        operand_outputs: &[f64],
        buf: Vec<f64>,
    ) -> Vec<f64>;

    fn regularization(&self, _parameter: f64) -> f64 {
        0.0
    }
}

#[derive(Debug, Clone, Copy)]
pub enum ComputationMode {
    Training,
    Inference,
}

#[derive(Debug)]
pub enum NodeComputation {
    Scalar(Box<dyn NodeScalarComputation>),
    Batch(Box<dyn NodeBatchComputation>),
}
impl NodeBackpropagationComputation for NodeComputation {
    fn compute_gradient_of_this_at_operand(
        &self,
        parameters: &[f64],
        operand_outputs: &[f64],
        buf: Vec<f64>,
    ) -> Vec<f64> {
        match self {
            NodeComputation::Scalar(x) => {
                NodeBackpropagationComputation::compute_gradient_of_this_at_operand(
                    x.as_ref(),
                    parameters,
                    operand_outputs,
                    buf,
                )
            }
            NodeComputation::Batch(x) => {
                NodeBackpropagationComputation::compute_gradient_of_this_at_operand(
                    x.as_ref(),
                    parameters,
                    operand_outputs,
                    buf,
                )
            }
        }
    }

    fn compute_gradient_of_this_at_parameter(
        &self,
        parameters: &[f64],
        operand_outputs: &[f64],
        buf: Vec<f64>,
    ) -> Vec<f64> {
        match self {
            NodeComputation::Scalar(x) => {
                NodeBackpropagationComputation::compute_gradient_of_this_at_parameter(
                    x.as_ref(),
                    parameters,
                    operand_outputs,
                    buf,
                )
            }
            NodeComputation::Batch(x) => {
                NodeBackpropagationComputation::compute_gradient_of_this_at_parameter(
                    x.as_ref(),
                    parameters,
                    operand_outputs,
                    buf,
                )
            }
        }
    }
}
