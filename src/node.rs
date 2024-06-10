//! # Terminologies
//!
//! - $f$: the function represented by the node
//!   - "this" in code
//! - $z$: the functions represented by the operands (predecessors) of the node
//!   - outputs of those functions become the input of the node
//! - $w$: the tunable parameters of $f$
//! - $E$: the outmost function represented by the root node of the computation graph
//!   - "root" in code

use std::{cell::RefCell, collections::VecDeque, sync::Arc};

use thiserror::Error;

use crate::{param::SharedParams, reused_buf::ReusedBuffers};

pub type SharedNode = Arc<RefCell<Node>>;

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

/// The function of this node should be
/// ```math
/// f : \mathbb{R}^n \to \mathbb{R}
/// ```
#[derive(Debug)]
pub struct Node {
    parameters: SharedParams,
    operands: Vec<SharedNode>,
    successor_len: usize,
    batch_cache: Vec<Cache>,
    computation: Arc<dyn NodeComputation + Sync + Send>,

    is_in_bfs_queue: bool,
    buf: ReusedBuffers<f64>,
}

impl Node {
    fn check_rep(&self) {
        if !cfg!(debug_assertions) {
            return;
        }
        for cache in &self.batch_cache {
            match &cache {
                Cache::Evaluate(x) => {
                    assert_eq!(x.operand_outputs.len(), self.operands.len());
                }
                Cache::Backpropagate(_) => {}
            }
        }
    }

    pub fn new(
        operands: Vec<SharedNode>,
        computation: Arc<dyn NodeComputation + Sync + Send>,
        parameters: Arc<RefCell<Vec<f64>>>,
    ) -> Node {
        operands.iter().for_each(|operand| {
            let mut operand = operand.borrow_mut();
            operand.increment_successor_len();
        });
        let this = Self {
            parameters,
            operands,
            successor_len: 0,
            batch_cache: vec![],
            computation,
            is_in_bfs_queue: false,
            buf: ReusedBuffers::new(u8::MAX as _),
        };
        this.check_rep();
        this
    }

    pub fn evaluate_once(&mut self, inputs: &[f64], batch_index: usize) -> f64 {
        let none = self.batch_cache.len() == batch_index;
        let some = self.batch_cache.len() == batch_index + 1;
        assert!(none || some);

        if let Some(Cache::Evaluate(cache)) = self.batch_cache.get(batch_index) {
            return cache.output;
        }

        let mut operand_outputs = self.buf.take();
        operand_outputs.extend(self.operands.iter_mut().map(|operand| {
            let mut operand = operand.borrow_mut();
            operand.evaluate_once(inputs, batch_index)
        }));
        let output = {
            let parameters = self.parameters.borrow();
            self.computation
                .compute_output(&parameters, &operand_outputs, inputs)
        };

        self.batch_cache.push(Cache::Evaluate(EvaluateCache {
            output,
            operand_outputs,
        }));

        self.check_rep();
        output
    }

    pub fn do_gradient_descent_step(&mut self, step_size: f64) -> Result<(), GradientDescentError> {
        if self.batch_cache.is_empty() {
            return Err(GradientDescentError::NoEvaluationOutputCaches);
        };
        for cache in &mut self.batch_cache {
            let cache = cache.transit_to_backpropagate();
            if cache.gradient_of_root_at_this(self.successor_len).is_none() {
                return Err(
                    GradientDescentError::NotReceivingEnoughAddendsOfGradientFromSuccessors,
                );
            }
        }
        self.adjust_parameters(step_size);
        self.check_rep();
        Ok(())
    }

    fn increment_successor_len(&mut self) {
        self.successor_len += 1;
        self.check_rep();
    }

    fn adjust_parameters(&mut self, step_size: f64) {
        let batch_size = self.batch_cache.len();

        let mut parameters = self.parameters.borrow_mut();

        // Distribute addends of partial derivatives of root at operands to operands
        for batch_index in 0..batch_size {
            let buf = self.buf.take();
            let gradient_of_this_at_operand = self
                .gradient_of_this_at_operand(batch_index, &parameters, buf)
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
            self.buf.put(gradient_of_this_at_operand);
        }

        let mut partial_derivative_of_root_at_parameter = self.buf.take();
        partial_derivative_of_root_at_parameter
            .extend(core::iter::repeat(0.).take(parameters.len()));
        for batch_index in 0..batch_size {
            let buf = self.buf.take();
            let gradient_of_root_at_parameter = self
                .gradient_of_root_at_parameter(batch_index, &parameters, buf)
                .unwrap();
            gradient_of_root_at_parameter.iter().enumerate().for_each(
                |(i, partial_derivative_of_root_at_parameter_i)| {
                    partial_derivative_of_root_at_parameter[i] +=
                        partial_derivative_of_root_at_parameter_i / (batch_size as f64);
                },
            );
            self.buf.put(gradient_of_root_at_parameter);
        }
        for (i, partial_derivative_of_root_at_parameter_i) in
            partial_derivative_of_root_at_parameter
                .iter()
                .copied()
                .enumerate()
        {
            let regularization = self.computation.regularization(parameters[i]);
            parameters[i] -=
                step_size * (partial_derivative_of_root_at_parameter_i + regularization);
        }
        self.buf.put(partial_derivative_of_root_at_parameter);
        // Clear batch cache
        while let Some(cache) = self.batch_cache.pop() {
            cache.put_buf(&mut self.buf);
        }
        self.check_rep();
    }

    fn add_addend_of_partial_derivative_of_root_at_this(
        &mut self,
        addend: f64,
        batch_index: usize,
    ) {
        let cache = self.batch_cache.get_mut(batch_index).unwrap();
        let cache = cache.transit_to_backpropagate();
        cache.add_up(addend);
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
        buf: Vec<f64>,
    ) -> Result<Vec<f64>, GradientOfThisAtOperandError> {
        let operand_outputs = self
            .operand_outputs(batch_index)
            .ok_or(GradientOfThisAtOperandError::NoEvaluationOutputCaches)?;
        Ok(self
            .computation
            .compute_gradient_of_this_at_operand(parameters, operand_outputs, buf))
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
        let Some(cache) = self.batch_cache.get(batch_index) else {
            return Err(
                GradientOfRootAtThisError::NotReceivingEnoughAddendsOfGradientFromSuccessors,
            );
        };

        let gradient_of_root_at_this = match &cache {
            Cache::Evaluate(_) => None,
            Cache::Backpropagate(x) => x.gradient_of_root_at_this(self.successor_len),
        };

        Ok(match gradient_of_root_at_this {
            Some(x) => x,
            None => {
                return Err(
                    GradientOfRootAtThisError::NotReceivingEnoughAddendsOfGradientFromSuccessors,
                );
            }
        })
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
        buf: Vec<f64>,
    ) -> Result<Vec<f64>, GradientOfThisAtParameterError> {
        let operand_outputs = self
            .operand_outputs(batch_index)
            .ok_or(GradientOfThisAtParameterError::NoEvaluationOutputCaches)?;
        Ok(self.computation.compute_gradient_of_this_at_parameter(
            parameters,
            operand_outputs.as_ref(),
            buf,
        ))
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
        buf: Vec<f64>,
    ) -> Result<Vec<f64>, GradientOfRootAtParameterError> {
        let gradient_of_this_at_parameter = self
            .gradient_of_this_at_parameter(batch_index, parameters, buf)
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
        Some(match &self.batch_cache.get(batch_index) {
            Some(Cache::Evaluate(x)) => &x.operand_outputs,
            Some(Cache::Backpropagate(x)) => &x.evaluate_cache.operand_outputs,
            None => return None,
        })
    }

    pub fn output(&self, batch_index: usize) -> Option<f64> {
        Some(match &self.batch_cache.get(batch_index) {
            Some(Cache::Evaluate(x)) => x.output,
            Some(Cache::Backpropagate(x)) => x.evaluate_cache.output,
            None => return None,
        })
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
        if n.batch_cache.is_empty() {
            return false;
        }
        n.batch_cache.clear();
        true
    };
    bfs_operands(root_note, f);
}

pub fn graph_do_gradient_descent_steps(root_note: &SharedNode, step_size: f64) {
    let f = |n: &mut Node| match n.do_gradient_descent_step(step_size) {
        Ok(_) => true,
        Err(e) => match e {
            GradientDescentError::NotReceivingEnoughAddendsOfGradientFromSuccessors => panic!(),
            // This node has had it parameters updated already
            GradientDescentError::NoEvaluationOutputCaches => false,
        },
    };
    bfs_operands(root_note, f);
}

/// `f`: Return `false` to trim this branch
fn bfs_operands(root_node: &SharedNode, f: impl Fn(&mut Node) -> bool) {
    let mut q = VecDeque::new();
    q.push_back(Arc::clone(root_node));

    while let Some(n) = q.pop_front() {
        let mut n = n.borrow_mut();
        n.set_is_in_bfs_queue(false);
        let should_visit_children = f(&mut n);
        if !should_visit_children {
            continue;
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
enum Cache {
    Evaluate(EvaluateCache),
    Backpropagate(BackpropagateCache),
}
impl Cache {
    pub fn transit_to_backpropagate(&mut self) -> &mut BackpropagateCache {
        match self {
            Cache::Evaluate(x) => {
                let evaluate_cache = core::mem::replace(
                    x,
                    EvaluateCache {
                        output: Default::default(),
                        operand_outputs: Default::default(),
                    },
                );
                *self = Cache::Backpropagate(BackpropagateCache::new(evaluate_cache));
                let Cache::Backpropagate(x) = self else {
                    unreachable!()
                };
                x
            }
            Cache::Backpropagate(x) => x,
        }
    }

    pub fn put_buf(self, buf: &mut ReusedBuffers<f64>) {
        match self {
            Cache::Evaluate(x) => {
                buf.put(x.operand_outputs);
            }
            Cache::Backpropagate(x) => {
                x.put_buf(buf);
            }
        }
    }
}

#[derive(Debug, Clone)]
struct EvaluateCache {
    /// the output of this node
    pub output: f64,

    /// the outputs of the operands
    pub operand_outputs: Vec<f64>,
}

#[derive(Debug, Clone)]
struct BackpropagateCache {
    sum_gradient_of_root_at_this: f64,
    times: usize,

    evaluate_cache: EvaluateCache,
}
impl BackpropagateCache {
    pub fn new(evaluate_cache: EvaluateCache) -> Self {
        Self {
            sum_gradient_of_root_at_this: 0.,
            times: 0,
            evaluate_cache,
        }
    }

    pub fn add_up(&mut self, addend: f64) {
        self.sum_gradient_of_root_at_this += addend;
        self.times += 1;
    }

    /// ```math
    /// (\frac{\partial E}{\partial h_i} \cdot \frac{\partial h_i}{\partial f})
    /// ```
    ///
    /// - $h_i$: the $i$-th immediate successor of this node
    /// - $f$: this node
    pub fn gradient_of_root_at_this(&self, successor_len: usize) -> Option<f64> {
        assert!(self.times <= successor_len);
        if successor_len == 0 {
            // this is the root node
            return Some(1.);
        }
        if self.times < successor_len {
            return None;
        }
        Some(self.sum_gradient_of_root_at_this)
    }

    pub fn put_buf(self, buf: &mut ReusedBuffers<f64>) {
        buf.put(self.evaluate_cache.operand_outputs);
    }
}
