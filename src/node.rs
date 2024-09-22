//! # Terminologies
//!
//! - $f$: the function represented by the node
//!   - "this" in code
//! - $z$: the functions represented by the operands (predecessors) of the node
//!   - outputs of those functions become the input of the node
//! - $w$: the tunable parameters of $f$
//! - $E$: the outmost function represented by the root node of the computation graph
//!   - "root" in code

use graph::{Graph, Node, NodeArray, NodeIdx};
use primitive::{
    obj_pool::{buf_pool, ObjectPool},
    vec_seg::{SegKey, VecSeg},
};
use thiserror::Error;

use crate::{
    cache::{GradRootThis, NodeCache, NodeCacheBuilder, OperandOutputs},
    computation::{ComputationMode, NodeBackpropagationComputation, NodeComputation},
    param::Params,
};

#[derive(Debug)]
pub struct NodeContext {
    buf: ObjectPool<Vec<f64>>,
}
impl NodeContext {
    pub fn new() -> Self {
        Self {
            buf: buf_pool(u16::MAX.into()),
        }
    }

    pub fn buf(&mut self) -> &mut ObjectPool<Vec<f64>> {
        &mut self.buf
    }
}
impl Default for NodeContext {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug)]
pub struct GraphBuilder {
    nodes: NodeArray<CompNode>,
}
impl GraphBuilder {
    pub fn new() -> Self {
        Self {
            nodes: NodeArray::with_key(),
        }
    }

    pub fn insert_nodes(&mut self, nodes: Vec<CompNode>) -> Vec<NodeIdx> {
        nodes.into_iter().map(|n| self.insert_node(n)).collect()
    }
    pub fn insert_node(&mut self, node: CompNode) -> NodeIdx {
        for &child in node.children() {
            self.nodes
                .get_mut(child)
                .unwrap()
                .increment_num_successors();
        }
        self.nodes.insert(node)
    }

    pub fn build(self) -> Graph<CompNode> {
        Graph::new(self.nodes)
    }
}
impl Default for GraphBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// The function of this node should be
/// ```math
/// f : \mathbb{R}^n \to \mathbb{R}
/// ```
#[derive(Debug)]
pub struct CompNode {
    parameters: SegKey,
    operands: Vec<NodeIdx>,
    num_successors: usize,
    batch_cache: Option<NodeCache>,
    computation: NodeComputation,
}
impl Node for CompNode {
    fn children(&self) -> &[NodeIdx] {
        &self.operands
    }
}
impl CompNode {
    fn check_rep(&self) {
        if !cfg!(debug_assertions) {}
    }

    pub fn new(
        operands: Vec<NodeIdx>,
        computation: NodeComputation,
        parameters: SegKey,
    ) -> CompNode {
        let this = Self {
            parameters,
            operands,
            num_successors: 0,
            batch_cache: None,
            computation,
        };
        this.check_rep();
        this
    }

    fn increment_num_successors(&mut self) {
        self.num_successors += 1;
        self.check_rep();
    }
    pub fn set_cache(&mut self, cache: NodeCache) {
        self.batch_cache = Some(cache);
    }
    pub fn clear_cache(&mut self, cx: &mut NodeContext) {
        self.batch_cache.take().unwrap().put_buf(cx);
    }
    pub fn has_evaluated(&self) -> bool {
        self.batch_cache.is_some()
    }

    pub fn evaluate_once<I>(
        &mut self,
        params: &mut Params,
        inputs_batch: &[I],
        operand_outputs: Vec<f64>,
        cx: &mut NodeContext,
        mode: ComputationMode,
    ) where
        I: AsRef<[f64]>,
    {
        assert!(!self.has_evaluated());

        let (mut cache_buf, operand_outputs) = VecSeg::from_vec(operand_outputs);
        let outputs_start = cache_buf.open_seg();

        // Compute outputs
        let output = match &mut self.computation {
            NodeComputation::Scalar(comp) => {
                let parameters = params.seg().slice(self.parameters);
                for (batch_index, inputs) in inputs_batch.iter().enumerate() {
                    let operand_outputs = OperandOutputs {
                        slice: cache_buf.slice(operand_outputs),
                        num_operands: self.operands.len(),
                    }
                    .get(batch_index);
                    let o = comp.compute_output(parameters, operand_outputs, inputs.as_ref());
                    if !o.is_finite() {
                        let e = format!(
                            "{comp:?}; output: {o}; params: {:?}; operands: {operand_outputs:?}; inputs: {:?}",
                            parameters,
                            inputs.as_ref(),
                        );
                        panic!("{e}");
                    }
                    cache_buf.push(o);
                }
                cache_buf.seal_seg(outputs_start)
            }
            NodeComputation::Batch(bat) => {
                let buf = cx.buf().take();
                let operand_outputs = cache_buf.slice(operand_outputs);
                let shape = &[self.operands.len(), inputs_batch.len()];
                let o =
                    bat.compute_output(params, self.parameters, operand_outputs, shape, buf, mode);
                let key = cache_buf.extend(o.iter().copied());
                cx.buf().put(o);
                key
            }
        };

        let cache = NodeCacheBuilder {
            batch_size: inputs_batch.len(),
            num_operands: self.operands.len(),
            buf: cache_buf,
            operand_outputs,
            output,
        }
        .build();
        self.batch_cache = Some(cache);
        self.check_rep();
    }

    pub fn is_ok_do_gradient_descent_step(&self) -> Result<(), GradientDescentError> {
        let Some(cache) = &self.batch_cache else {
            return Err(GradientDescentError::NoEvaluationOutputCaches);
        };
        if matches!(
            cache
                .backpropagate()
                .gradient_of_root_at_this(self.num_successors),
            GradRootThis::NoEnoughAddends
        ) {
            return Err(GradientDescentError::NotReceivingEnoughAddendsOfGradientFromSuccessors);
        }
        Ok(())
    }

    /// Distribute addends of partial derivatives of root at operands to operands
    pub fn distribute_gradient_to_operands(
        &self,
        params: &Params,
        buf: &mut Vec<(NodeIdx, usize, f64)>,
        cx: &mut NodeContext,
    ) {
        let batch_size = self.batch_cache.as_ref().unwrap().batch_size();
        let params = params.seg().slice(self.parameters);

        for batch_index in 0..batch_size {
            let gradient_of_this_at_operand = self
                .gradient_of_this_at_operand(batch_index, params, cx)
                .unwrap();
            let d_root_at_this = self
                .partial_derivative_of_root_at_this(batch_index)
                .unwrap();
            for (d_this_at_operand, operand) in
                gradient_of_this_at_operand.iter().zip(self.operands.iter())
            {
                // ```math
                // \frac{\partial E}{\partial f} \cdot \frac{\partial f}{\partial z}
                // ```
                let addend = d_root_at_this * d_this_at_operand;
                buf.push((*operand, batch_index, addend));
            }
            cx.buf().put(gradient_of_this_at_operand);
        }
    }

    pub fn adjust_parameters(&mut self, params: &mut Params, step_size: f64, cx: &mut NodeContext) {
        let batch_size = self.batch_cache.as_ref().unwrap().batch_size();
        let params = params.seg_mut().slice_mut(self.parameters);

        let mut partial_derivative_of_root_at_parameter = cx.buf().take();
        partial_derivative_of_root_at_parameter.extend(core::iter::repeat(0.).take(params.len()));
        for batch_index in 0..batch_size {
            let gradient_of_root_at_parameter = self
                .gradient_of_root_at_parameter(batch_index, params, cx)
                .unwrap();
            for (batch_sum, x) in partial_derivative_of_root_at_parameter
                .iter_mut()
                .zip(gradient_of_root_at_parameter.iter().copied())
            {
                *batch_sum += x / (batch_size as f64)
            }
            cx.buf().put(gradient_of_root_at_parameter);
        }
        for (param, der) in params
            .iter_mut()
            .zip(partial_derivative_of_root_at_parameter.iter().copied())
        {
            let regularization = self.computation.regularization(*param);
            *param -= step_size * (der + regularization);
        }
        cx.buf().put(partial_derivative_of_root_at_parameter);
        // Clear batch cache
        self.clear_cache(cx);
        self.check_rep();
    }

    pub fn add_addend_of_partial_derivative_of_root_at_this(
        &mut self,
        addend: f64,
        batch_index: usize,
    ) {
        let cache = self.batch_cache.as_mut().unwrap();
        cache.backpropagate_mut().add_up(addend, batch_index);
        self.check_rep();
    }

    /// ```math
    /// \frac{\partial f}{\partial z}
    /// ```
    ///
    /// - $z$: the non-tunable operands of this node
    /// - $f$: this node
    pub fn gradient_of_this_at_operand(
        &self,
        batch_index: usize,
        parameters: &[f64],
        cx: &mut NodeContext,
    ) -> Result<Vec<f64>, GradientOfThisAtOperandError> {
        let operand_outputs = self
            .operand_outputs(batch_index)
            .ok_or(GradientOfThisAtOperandError::NoEvaluationOutputCaches)?;
        let buf = cx.buf().take();
        let x =
            self.computation
                .compute_gradient_of_this_at_operand(parameters, operand_outputs, buf);
        Ok(x)
    }

    /// ```math
    /// \frac{\partial E}{\partial f}
    /// ```
    ///
    /// - $E$: the out-most function of the entire network
    /// - $f$: this node
    pub fn partial_derivative_of_root_at_this(
        &self,
        batch_index: usize,
    ) -> Result<f64, GradientOfRootAtThisError> {
        let Some(cache) = &self.batch_cache else {
            return Err(
                GradientOfRootAtThisError::NotReceivingEnoughAddendsOfGradientFromSuccessors,
            );
        };

        Ok(
            match cache
                .backpropagate()
                .gradient_of_root_at_this(self.num_successors)
            {
                GradRootThis::Some(x) => x[batch_index],
                GradRootThis::NoEnoughAddends => {
                    return Err(
                    GradientOfRootAtThisError::NotReceivingEnoughAddendsOfGradientFromSuccessors,
                    );
                }
                GradRootThis::AllOnes => 1.,
            },
        )
    }

    /// ```math
    /// \frac{\partial f}{\partial w}
    /// ```
    ///
    /// - $w$: the tunable parameters of this node
    /// - $f$: this node
    pub fn gradient_of_this_at_parameter(
        &self,
        batch_index: usize,
        parameters: &[f64],
        cx: &mut NodeContext,
    ) -> Result<Vec<f64>, GradientOfThisAtParameterError> {
        let operand_outputs = self
            .operand_outputs(batch_index)
            .ok_or(GradientOfThisAtParameterError::NoEvaluationOutputCaches)?;
        let buf = cx.buf().take();
        let x = self.computation.compute_gradient_of_this_at_parameter(
            parameters,
            operand_outputs.as_ref(),
            buf,
        );
        Ok(x)
    }

    /// ```math
    /// \frac{\partial E}{\partial w}
    /// ```
    ///
    /// - $w$: the tunable parameters of this node
    pub fn gradient_of_root_at_parameter(
        &self,
        batch_index: usize,
        parameters: &[f64],
        cx: &mut NodeContext,
    ) -> Result<Vec<f64>, GradientOfRootAtParameterError> {
        let gradient_of_this_at_parameter = self
            .gradient_of_this_at_parameter(batch_index, parameters, cx)
            .map_err(GradientOfRootAtParameterError::GradientOfThisAtParameter)?;
        let d_root_at_this = self
            .partial_derivative_of_root_at_this(batch_index)
            .map_err(GradientOfRootAtParameterError::GradientOfRootAtThis)?;
        let mut x = gradient_of_this_at_parameter;
        for x in &mut x {
            let d_this_at_param = *x;
            let d_root_at_param = d_root_at_this * d_this_at_param;
            *x = d_root_at_param;
        }
        Ok(x)
    }

    pub fn operand_outputs(&self, batch_index: usize) -> Option<&[f64]> {
        let cache = self.batch_cache.as_ref()?;
        Some(cache.operand_outputs(batch_index))
    }

    pub fn output(&self) -> Option<&[f64]> {
        let cache = self.batch_cache.as_ref()?;
        Some(cache.output())
    }

    pub fn parameters(&self) -> SegKey {
        self.parameters
    }
}

#[derive(Debug, Error)]
pub enum GradientDescentError {
    #[error("Not receiving enough addends of gradient of root node at this node from successors")]
    NotReceivingEnoughAddendsOfGradientFromSuccessors,
    #[error("No evaluation output caches")]
    NoEvaluationOutputCaches,
}

#[derive(Debug, Error)]
pub enum GradientOfRootAtThisError {
    #[error("Not receiving enough addends of gradient of root node at this node from successors")]
    NotReceivingEnoughAddendsOfGradientFromSuccessors,
}

#[derive(Debug, Error)]
pub enum GradientOfRootAtParameterError {
    #[error("Gradient of root node at this node error: {0}")]
    GradientOfRootAtThis(GradientOfRootAtThisError),
    #[error("Gradient of this node at parameter error: {0}")]
    GradientOfThisAtParameter(GradientOfThisAtParameterError),
}

#[derive(Debug, Error)]
pub enum GradientOfThisAtParameterError {
    #[error("No evaluation output caches")]
    NoEvaluationOutputCaches,
}

#[derive(Debug, Error)]
pub enum GradientOfThisAtOperandError {
    #[error("No evaluation output caches")]
    NoEvaluationOutputCaches,
}

pub fn evaluate_once<I>(
    graph: &mut Graph<CompNode>,
    nodes_forward: &[NodeIdx],
    params: &mut Params,
    inputs_batch: &[I],
    cx: &mut NodeContext,
    mode: ComputationMode,
) where
    I: AsRef<[f64]>,
{
    let mut buf = vec![];
    for &node in nodes_forward {
        {
            let node = graph.nodes().get(node).unwrap();
            if node.has_evaluated() {
                continue;
            }
            buf.clear();
            buf.extend(node.children());
        }

        // Collect operand outputs
        let mut operand_outputs = cx.buf().take();
        for batch_index in 0..inputs_batch.len() {
            for &operand in &buf {
                let operand = graph.nodes().get(operand).unwrap();
                operand_outputs.push(operand.output().unwrap()[batch_index])
            }
        }

        let node = graph.nodes_mut().get_mut(node).unwrap();
        node.evaluate_once(params, inputs_batch, operand_outputs, cx, mode);
    }
}
pub fn delete_cache(graph: &mut Graph<CompNode>, nodes_forward: &[NodeIdx], cx: &mut NodeContext) {
    for &node in nodes_forward {
        graph.nodes_mut().get_mut(node).unwrap().clear_cache(cx);
    }
}

pub fn backpropagate(
    graph: &mut Graph<CompNode>,
    nodes_forward: &[NodeIdx],
    params: &mut Params,
    step_size: f64,
    cx: &mut NodeContext,
) {
    let mut buf = vec![];
    for &node in nodes_forward.iter().rev() {
        match graph
            .nodes()
            .get(node)
            .unwrap()
            .is_ok_do_gradient_descent_step()
        {
            Ok(()) => (),
            Err(GradientDescentError::NotReceivingEnoughAddendsOfGradientFromSuccessors) => {
                panic!();
            }
            Err(GradientDescentError::NoEvaluationOutputCaches) => {
                // This node has had it parameters updated already
                panic!();
            }
        }
        buf.clear();
        graph
            .nodes()
            .get(node)
            .unwrap()
            .distribute_gradient_to_operands(params, &mut buf, cx);
        for &(child, batch_index, addend) in &buf {
            graph
                .nodes_mut()
                .get_mut(child)
                .unwrap()
                .add_addend_of_partial_derivative_of_root_at_this(addend, batch_index);
        }
        graph
            .nodes_mut()
            .get_mut(node)
            .unwrap()
            .adjust_parameters(params, step_size, cx);
    }
}
