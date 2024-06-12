//! # Terminologies
//!
//! - $f$: the function represented by the node
//!   - "this" in code
//! - $z$: the functions represented by the operands (predecessors) of the node
//!   - outputs of those functions become the input of the node
//! - $w$: the tunable parameters of $f$
//! - $E$: the outmost function represented by the root node of the computation graph
//!   - "root" in code

use std::{collections::VecDeque, sync::Arc};

use thiserror::Error;

use crate::{mut_cell::MutCell, param::SharedParams, reused_buf::ReusedBuffers};

pub type SharedNode = Arc<MutCell<Node>>;

/// The function of this node should be
/// ```math
/// f : \mathbb{R}^n \to \mathbb{R}
/// ```
pub trait NodeComputation: core::fmt::Debug {
    fn compute_output(
        &self,
        parameters: &[f64],
        operand_outputs: &[f64],
        graph_inputs: &[f64],
    ) -> f64;

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

#[derive(Debug)]
pub struct NodeContext {
    buf: ReusedBuffers<f64>,
    buf_buf: ReusedBuffers<Vec<f64>>,
}
impl NodeContext {
    pub fn new() -> Self {
        Self {
            buf: ReusedBuffers::new(u16::MAX.into()),
            buf_buf: ReusedBuffers::new(u16::MAX.into()),
        }
    }

    pub fn buf(&mut self) -> &mut ReusedBuffers<f64> {
        &mut self.buf
    }
    pub fn buf_buf(&mut self) -> &mut ReusedBuffers<Vec<f64>> {
        &mut self.buf_buf
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
    batch_cache: Option<Cache>,
    computation: Arc<dyn NodeComputation + Sync + Send>,

    is_in_bfs_queue: bool,
}

impl Node {
    fn check_rep(&self) {
        if !cfg!(debug_assertions) {
            return;
        }
        if let Some(cache) = &self.batch_cache {
            assert_eq!(
                cache.eval().output.len(),
                cache.backpropagate().sum_gradient_of_root_at_this.len()
            );
        }
    }

    pub fn new(
        operands: Vec<SharedNode>,
        computation: Arc<dyn NodeComputation + Sync + Send>,
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

        let mut operand_outputs_batch = cx.buf_buf().take();
        for batch_index in 0..inputs_batch.len() {
            let mut operand_outputs = cx.buf().take();
            for operand in &self.operands {
                let operand = operand.as_ref().borrow();
                operand_outputs.push(operand.output().unwrap()[batch_index])
            }
            operand_outputs_batch.push(operand_outputs);
        }

        let mut output = cx.buf().take();
        for (inputs, operand_outputs) in inputs_batch.iter().zip(&operand_outputs_batch) {
            let parameters = self.parameters.as_ref().borrow();
            let o = self
                .computation
                .compute_output(&parameters, operand_outputs, inputs.as_ref());
            output.push(o)
        }

        self.batch_cache = Some(Cache::new(
            EvaluateCache {
                output,
                operand_outputs: operand_outputs_batch,
            },
            cx.buf().take(),
        ));
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
            let partial_derivative_of_root_at_this = self
                .partial_derivative_of_root_at_this(batch_index)
                .unwrap();
            for (i, operand) in self.operands.iter().enumerate() {
                // ```math
                // \frac{\partial E}{\partial f} \cdot \frac{\partial f}{\partial z}
                // ```
                let addend_of_partial_derivative_of_root_at_operand =
                    partial_derivative_of_root_at_this * gradient_of_this_at_operand[i];
                let mut operand = operand.borrow_mut();
                operand.add_addend_of_partial_derivative_of_root_at_this(
                    addend_of_partial_derivative_of_root_at_operand,
                    batch_index,
                );
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
        for (param, partial_derivative_of_root_at_parameter_i) in parameters
            .iter_mut()
            .zip(partial_derivative_of_root_at_parameter.iter().copied())
        {
            let regularization = self.computation.regularization(*param);
            *param -= step_size * (partial_derivative_of_root_at_parameter_i + regularization);
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
        let partial_derivative_of_root_at_this = self
            .partial_derivative_of_root_at_this(batch_index)
            .map_err(GradientOfRootAtParameterError::GradientOfRootAtThis)?;
        let mut gradient_of_root_at_parameter = gradient_of_this_at_parameter;
        gradient_of_root_at_parameter.iter_mut().for_each(
            |partial_derivative_of_this_at_parameter_i| {
                *partial_derivative_of_this_at_parameter_i *= partial_derivative_of_root_at_this
            },
        );
        Ok(gradient_of_root_at_parameter)
    }

    pub fn operand_outputs(&self, batch_index: usize) -> Option<&Vec<f64>> {
        let cache = self.batch_cache.as_ref()?;
        Some(&cache.eval().operand_outputs[batch_index])
    }

    pub fn output(&self) -> Option<&Vec<f64>> {
        let cache = self.batch_cache.as_ref()?;
        Some(&cache.eval().output)
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

#[derive(Debug, Clone)]
struct Cache {
    eval: EvaluateCache,
    backpropagate: BackpropagateCache,
}
impl Cache {
    pub fn new(eval: EvaluateCache, buf: Vec<f64>) -> Self {
        let batch_size = eval.output.len();
        Self {
            eval,
            backpropagate: BackpropagateCache::new(batch_size, buf),
        }
    }

    pub fn put_buf(mut self, cx: &mut NodeContext) {
        cx.buf().put(self.eval.output);
        while let Some(o) = self.eval.operand_outputs.pop() {
            cx.buf().put(o);
        }
        cx.buf_buf().put(self.eval.operand_outputs);
        cx.buf()
            .put(self.backpropagate.sum_gradient_of_root_at_this);
    }

    pub fn eval(&self) -> &EvaluateCache {
        &self.eval
    }
    pub fn backpropagate(&self) -> &BackpropagateCache {
        &self.backpropagate
    }
    pub fn backpropagate_mut(&mut self) -> &mut BackpropagateCache {
        &mut self.backpropagate
    }
    pub fn batch_size(&self) -> usize {
        self.eval.output.len()
    }
}

#[derive(Debug, Clone)]
struct EvaluateCache {
    /// the output of this node
    pub output: Vec<f64>,
    /// the output from operands
    pub operand_outputs: Vec<Vec<f64>>,
}

#[derive(Debug, Clone)]
struct BackpropagateCache {
    sum_gradient_of_root_at_this: Vec<f64>,
    times: usize,
}
impl BackpropagateCache {
    pub fn new(batch_size: usize, mut buf: Vec<f64>) -> Self {
        buf.extend(core::iter::repeat(0.).take(batch_size));
        Self {
            sum_gradient_of_root_at_this: buf,
            times: 0,
        }
    }

    pub fn add_up(&mut self, addend: f64, batch_index: usize) {
        self.sum_gradient_of_root_at_this[batch_index] += addend;
        if batch_index + 1 == self.sum_gradient_of_root_at_this.len() {
            self.times += 1;
        }
    }

    /// ```math
    /// (\frac{\partial E}{\partial h_i} \cdot \frac{\partial h_i}{\partial f})
    /// ```
    ///
    /// - $h_i$: the $i$-th immediate successor of this node
    /// - $f$: this node
    pub fn gradient_of_root_at_this(&self, num_successors: usize) -> GradRootThis<'_> {
        assert!(self.times <= num_successors);
        if num_successors == 0 {
            // this is the root node
            return GradRootThis::AllOnes;
        }
        if self.times < num_successors {
            return GradRootThis::NoEnoughAddends;
        }
        GradRootThis::Some(&self.sum_gradient_of_root_at_this)
    }
}

enum GradRootThis<'a> {
    AllOnes,
    Some(&'a [f64]),
    NoEnoughAddends,
}
