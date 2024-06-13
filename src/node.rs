//! # Terminologies
//!
//! - $f$: the function represented by the node
//!   - "this" in code
//! - $z$: the functions represented by the operands (predecessors) of the node
//!   - outputs of those functions become the input of the node
//! - $w$: the tunable parameters of $f$
//! - $E$: the outmost function represented by the root node of the computation graph
//!   - "root" in code

use std::{collections::VecDeque, ops::DerefMut, sync::Arc};

use thiserror::Error;

use crate::{
    cache::{GradRootThis, NodeCache, NodeCacheBuilder, OperandOutputs},
    computation::{NodeBackpropagationComputation, NodeComputation},
    mut_cell::MutCell,
    param::SharedParams,
    reused_buf::ReusedBuffers,
};

pub type SharedNode = Arc<MutCell<Node>>;

#[derive(Debug)]
pub struct NodeContext {
    buf: ReusedBuffers<f64>,
}
impl NodeContext {
    pub fn new() -> Self {
        Self {
            buf: ReusedBuffers::new(u16::MAX.into()),
        }
    }

    pub fn buf(&mut self) -> &mut ReusedBuffers<f64> {
        &mut self.buf
    }
}
impl Default for NodeContext {
    fn default() -> Self {
        Self::new()
    }
}

/// The function of this node should be
/// ```math
/// f : \mathbb{R}^n \to \mathbb{R}
/// ```
#[derive(Debug)]
pub struct Node {
    parameters: SharedParams,
    operands: Vec<SharedNode>,
    num_successors: usize,
    batch_cache: Option<NodeCache>,
    computation: Arc<MutCell<NodeComputation>>,

    is_in_bfs_queue: bool,
}

impl Node {
    fn check_rep(&self) {
        if !cfg!(debug_assertions) {}
    }

    pub fn new(
        operands: Vec<SharedNode>,
        computation: Arc<MutCell<NodeComputation>>,
        parameters: Arc<MutCell<Vec<f64>>>,
    ) -> Node {
        operands.iter().for_each(|operand| {
            let mut operand = operand.borrow_mut();
            operand.increment_num_successors();
        });
        let this = Self {
            parameters,
            operands,
            num_successors: 0,
            batch_cache: None,
            computation,
            is_in_bfs_queue: false,
        };
        this.check_rep();
        this
    }

    pub fn evaluate_once<I>(&mut self, inputs_batch: &[I], cx: &mut NodeContext)
    where
        I: AsRef<[f64]>,
    {
        if self.batch_cache.is_some() {
            return;
        }

        for operand in &self.operands {
            let mut operand = operand.borrow_mut();
            operand.evaluate_once(inputs_batch, cx);
        }

        // Collect operand outputs
        let mut eval_buf = cx.buf().take();
        for batch_index in 0..inputs_batch.len() {
            for operand in &self.operands {
                let operand = operand.as_ref().borrow();
                eval_buf.push(operand.output().unwrap()[batch_index])
            }
        }
        let operand_outputs_len = eval_buf.len();

        // Compute outputs
        let mut computation = self.computation.as_ref().borrow_mut();
        match computation.deref_mut() {
            NodeComputation::Scalar(comp) => {
                for (batch_index, inputs) in inputs_batch.iter().enumerate() {
                    let operand_outputs = OperandOutputs {
                        slice: &eval_buf[..operand_outputs_len],
                        num_operands: self.operands.len(),
                    }
                    .get(batch_index);
                    let parameters = self.parameters.as_ref().borrow();
                    let o = comp.compute_output(&parameters, operand_outputs, inputs.as_ref());
                    eval_buf.push(o);
                }
            }
            NodeComputation::Batch(bat) => {
                let buf = cx.buf().take();
                let parameters = self.parameters.as_ref().borrow();
                let operand_outputs = &eval_buf;
                let shape = &[self.operands.len(), inputs_batch.len()];
                let o = bat.compute_output(&parameters, operand_outputs, shape, buf);
                eval_buf.extend(o.iter().copied());
                cx.buf().put(o);
            }
        }

        let cache = NodeCacheBuilder {
            batch_size: inputs_batch.len(),
            num_operands: self.operands.len(),
            buf: eval_buf,
        }
        .build();
        self.batch_cache = Some(cache);
        self.check_rep();
    }

    pub fn do_gradient_descent_step(
        &mut self,
        step_size: f64,
        cx: &mut NodeContext,
    ) -> Result<(), GradientDescentError> {
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
        self.adjust_parameters(step_size, cx);
        self.check_rep();
        Ok(())
    }

    fn increment_num_successors(&mut self) {
        self.num_successors += 1;
        self.check_rep();
    }

    fn adjust_parameters(&mut self, step_size: f64, cx: &mut NodeContext) {
        let batch_size = self.batch_cache.as_ref().unwrap().batch_size();

        let mut parameters = self.parameters.borrow_mut();

        // Distribute addends of partial derivatives of root at operands to operands
        for batch_index in 0..batch_size {
            let gradient_of_this_at_operand = self
                .gradient_of_this_at_operand(batch_index, &parameters, cx)
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
                let mut operand = operand.borrow_mut();
                operand.add_addend_of_partial_derivative_of_root_at_this(addend, batch_index);
            }
            cx.buf().put(gradient_of_this_at_operand);
        }

        let mut partial_derivative_of_root_at_parameter = cx.buf().take();
        partial_derivative_of_root_at_parameter
            .extend(core::iter::repeat(0.).take(parameters.len()));
        for batch_index in 0..batch_size {
            let gradient_of_root_at_parameter = self
                .gradient_of_root_at_parameter(batch_index, &parameters, cx)
                .unwrap();
            for (batch_sum, x) in partial_derivative_of_root_at_parameter
                .iter_mut()
                .zip(gradient_of_root_at_parameter.iter().copied())
            {
                *batch_sum += x / (batch_size as f64)
            }
            cx.buf().put(gradient_of_root_at_parameter);
        }
        for (param, der) in parameters
            .iter_mut()
            .zip(partial_derivative_of_root_at_parameter.iter().copied())
        {
            let regularization = self.computation.borrow().regularization(*param);
            *param -= step_size * (der + regularization);
        }
        cx.buf().put(partial_derivative_of_root_at_parameter);
        // Clear batch cache
        self.batch_cache.take().unwrap().put_buf(cx);
        self.check_rep();
    }

    fn add_addend_of_partial_derivative_of_root_at_this(
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
        let x = self
            .computation
            .borrow()
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
        let x = self
            .computation
            .borrow()
            .compute_gradient_of_this_at_parameter(parameters, operand_outputs.as_ref(), buf);
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

    pub fn parameters(&self) -> &SharedParams {
        &self.parameters
    }
    pub fn is_in_bfs_queue(&self) -> bool {
        self.is_in_bfs_queue
    }
    pub fn set_is_in_bfs_queue(&mut self, value: bool) {
        self.is_in_bfs_queue = value;
    }
}

pub fn clone_node_batch(nodes: &[SharedNode]) -> Vec<SharedNode> {
    nodes.iter().map(Arc::clone).collect()
}

pub fn graph_delete_caches(root_note: &SharedNode) {
    let f = |n: &mut Node| {
        if n.batch_cache.is_none() {
            return BfsNextMove::Noop;
        }
        n.batch_cache = None;
        BfsNextMove::VisitChildren
    };
    bfs_operands(root_note, f);
}

pub fn graph_do_gradient_descent_steps(
    root_note: &SharedNode,
    step_size: f64,
    cx: &mut NodeContext,
) {
    let f = |n: &mut Node| match n.do_gradient_descent_step(step_size, cx) {
        Ok(_) => BfsNextMove::VisitChildren,
        Err(e) => match e {
            GradientDescentError::NotReceivingEnoughAddendsOfGradientFromSuccessors => {
                BfsNextMove::Reschedule
            }
            // This node has had it parameters updated already
            GradientDescentError::NoEvaluationOutputCaches => BfsNextMove::Noop,
        },
    };
    bfs_operands(root_note, f);
}

fn bfs_operands<V>(root_node: &SharedNode, mut visit: V)
where
    V: FnMut(&mut Node) -> BfsNextMove,
{
    let mut q = VecDeque::new();
    q.push_back(Arc::clone(root_node));

    while let Some(node) = q.pop_front() {
        let mut n = node.borrow_mut();
        n.set_is_in_bfs_queue(false);
        let next_move = visit(&mut n);
        match next_move {
            BfsNextMove::Reschedule => {
                drop(n);
                q.push_back(node);
                continue;
            }
            BfsNextMove::Noop => continue,
            BfsNextMove::VisitChildren => (),
        }
        for op in &n.operands {
            {
                let mut op = op.borrow_mut();
                if op.is_in_bfs_queue() {
                    continue;
                }
                op.set_is_in_bfs_queue(true);
            }
            q.push_back(Arc::clone(op));
        }
    }
}
enum BfsNextMove {
    /// Put self back to the queue
    Reschedule,
    Noop,
    VisitChildren,
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
