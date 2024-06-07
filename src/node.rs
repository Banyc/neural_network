//! # Terminologies
//!
//! - $f$: the function represented by the node
//!   - "this" in code
//! - $z$: the functions represented by the operands (predecessors) of the node
//!   - outputs of those functions become the input of the node
//! - $w$: the tunable parameters of $f$
//! - $E$: the outmost function represented by the root node of the computation graph
//!   - "root" in code

use std::{
    collections::VecDeque,
    sync::{Arc, Mutex},
};

use thiserror::Error;

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
    parameters: Vec<f64>,
    operands: Vec<Arc<Mutex<Node>>>,
    successor_len: usize,
    batch_cache: Vec<Cache>,
    computation: Arc<dyn NodeComputation + Sync + Send>,
}

impl Node {
    fn check_rep(&self) {
        for cache in &self.batch_cache {
            match &cache {
                Cache::Evaluate(x) => {
                    assert_eq!(x.operand_outputs.len(), self.operands.len());
                }
                Cache::Backpropagate(x) => {
                    assert!(x.addends_of_gradient_of_root_at_this.len() <= self.successor_len);
                }
            }
        }
    }

    pub fn new(
        operands: Vec<Arc<Mutex<Node>>>,
        computation: Arc<dyn NodeComputation + Sync + Send>,
        parameters: Vec<f64>,
    ) -> Node {
        operands.iter().for_each(|operand| {
            let mut operand = operand.lock().unwrap();
            operand.increment_successor_len();
        });
        let this = Self {
            parameters,
            operands,
            successor_len: 0,
            batch_cache: vec![],
            computation,
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

        let operand_outputs: Vec<f64> = self
            .operands
            .iter_mut()
            .map(|operand| {
                let mut operand = operand.lock().unwrap();
                operand.evaluate_once(inputs, batch_index)
            })
            .collect();
        let output = self
            .computation
            .compute_output(&self.parameters, &operand_outputs, inputs);

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
        for cache in &self.batch_cache {
            match cache {
                Cache::Evaluate(_) => {
                    if self.successor_len != 0 {
                        return Err(
                            GradientDescentError::NotReceivingEnoughAddendsOfGradientFromSuccessors,
                        );
                    }
                }
                Cache::Backpropagate(x) => {
                    if self.successor_len > x.addends_of_gradient_of_root_at_this.len() {
                        return Err(
                            GradientDescentError::NotReceivingEnoughAddendsOfGradientFromSuccessors,
                        );
                    }
                }
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

        // Distribute addends of partial derivatives of root at operands to operands
        for batch_index in 0..batch_size {
            let gradient_of_this_at_operand =
                self.gradient_of_this_at_operand(batch_index).unwrap();
            let partial_derivative_of_root_at_this = self
                .partial_derivative_of_root_at_this(batch_index)
                .unwrap();
            (0..self.operands.len()).for_each(|i| {
                // ```math
                // \frac{\partial E}{\partial f} \cdot \frac{\partial f}{\partial z}
                // ```
                let addend_of_partial_derivative_of_root_at_operand =
                    partial_derivative_of_root_at_this * gradient_of_this_at_operand[i];
                let mut operand = self.operands[i].lock().unwrap();
                operand.add_addend_of_partial_derivative_of_root_at_this(
                    addend_of_partial_derivative_of_root_at_operand,
                    batch_index,
                );
            });
        }

        let mut partial_derivative_of_root_at_parameter = vec![0.; self.parameters.len()];
        for batch_index in 0..batch_size {
            let gradient_of_root_at_parameter =
                self.gradient_of_root_at_parameter(batch_index).unwrap();
            gradient_of_root_at_parameter.iter().enumerate().for_each(
                |(i, partial_derivative_of_root_at_parameter_i)| {
                    partial_derivative_of_root_at_parameter[i] +=
                        partial_derivative_of_root_at_parameter_i / (batch_size as f64);
                },
            );
        }
        for (i, partial_derivative_of_root_at_parameter_i) in
            partial_derivative_of_root_at_parameter
                .into_iter()
                .enumerate()
        {
            let regularization = self.computation.regularization(self.parameters[i]);
            self.parameters[i] -=
                step_size * (partial_derivative_of_root_at_parameter_i + regularization);
        }
        self.batch_cache.clear();
        self.check_rep();
    }

    fn add_addend_of_partial_derivative_of_root_at_this(
        &mut self,
        addend: f64,
        batch_index: usize,
    ) {
        let cache = self.batch_cache.get_mut(batch_index).unwrap();
        let cache = match cache {
            Cache::Evaluate(x) => {
                *self.batch_cache.get_mut(batch_index).unwrap() =
                    Cache::Backpropagate(BackpropagateCache {
                        addends_of_gradient_of_root_at_this: vec![],
                        evaluate_cache: x.clone(),
                    });
                self.add_addend_of_partial_derivative_of_root_at_this(addend, batch_index);
                return;
            }
            Cache::Backpropagate(x) => x,
        };
        cache.addends_of_gradient_of_root_at_this.push(addend);
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
    ) -> Result<Vec<f64>, GradientOfThisAtOperandError> {
        let operand_outputs = self
            .operand_outputs(batch_index)
            .ok_or(GradientOfThisAtOperandError::NoEvaluationOutputCaches)?;
        Ok(self
            .computation
            .compute_gradient_of_this_at_operand(&self.parameters, operand_outputs))
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

        let addends_of_gradient_of_root_at_this = match &cache {
            Cache::Evaluate(_) => &[],
            Cache::Backpropagate(x) => x.addends_of_gradient_of_root_at_this.as_slice(),
        };

        if self.successor_len != addends_of_gradient_of_root_at_this.len() {
            return Err(
                GradientOfRootAtThisError::NotReceivingEnoughAddendsOfGradientFromSuccessors,
            );
        }

        Ok(if self.successor_len == 0 {
            // this is the root node
            1.0
        } else {
            addends_of_gradient_of_root_at_this.iter().sum()
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
    ) -> Result<Vec<f64>, GradientOfThisAtParameterError> {
        let operand_outputs = self
            .operand_outputs(batch_index)
            .ok_or(GradientOfThisAtParameterError::NoEvaluationOutputCaches)?;
        Ok(self
            .computation
            .compute_gradient_of_this_at_parameter(&self.parameters, operand_outputs.as_ref()))
    }

    /// ```math
    /// \frac{\partial E}{\partial w}
    /// ```
    ///
    /// - $w$: the tunable parameters of this node
    pub fn gradient_of_root_at_parameter(
        &self,
        batch_index: usize,
    ) -> Result<Vec<f64>, GradientOfRootAtParameterError> {
        let gradient_of_this_at_parameter = self
            .gradient_of_this_at_parameter(batch_index)
            .map_err(GradientOfRootAtParameterError::GradientOfThisAtParameter)?;
        let partial_derivative_of_root_at_this = self
            .partial_derivative_of_root_at_this(batch_index)
            .map_err(GradientOfRootAtParameterError::GradientOfRootAtThis)?;
        Ok(gradient_of_this_at_parameter
            .iter()
            .map(|partial_derivative_of_this_at_parameter_i| {
                partial_derivative_of_root_at_this * *partial_derivative_of_this_at_parameter_i
            })
            .collect())
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

    pub fn parameters(&self) -> &Vec<f64> {
        &self.parameters
    }
    pub fn set_parameters(&mut self, parameters: Vec<f64>) {
        assert_eq!(self.parameters.len(), parameters.len());
        self.parameters = parameters;
    }
}

pub fn clone_node_batch(nodes: &[Arc<Mutex<Node>>]) -> Vec<Arc<Mutex<Node>>> {
    nodes.iter().map(Arc::clone).collect()
}

pub fn graph_delete_caches(root_note: &Arc<Mutex<Node>>) {
    let f = |n: &mut Node| {
        if n.batch_cache.is_empty() {
            return false;
        }
        n.batch_cache.clear();
        true
    };
    bfs_operands(root_note, f);
}

pub fn graph_do_gradient_descent_steps(root_note: &Arc<Mutex<Node>>, step_size: f64) {
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
fn bfs_operands(root_node: &Arc<Mutex<Node>>, f: impl Fn(&mut Node) -> bool) {
    let mut q = VecDeque::new();
    q.push_back(Arc::clone(root_node));

    while let Some(n) = q.pop_front() {
        let mut n = n.lock().unwrap();
        let should_visit_children = f(&mut n);
        if !should_visit_children {
            continue;
        }
        for op in &n.operands {
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

#[derive(Debug, Clone)]
struct EvaluateCache {
    /// the output of this node
    pub output: f64,

    /// the outputs of the operands
    pub operand_outputs: Vec<f64>,
}

#[derive(Debug, Clone)]
struct BackpropagateCache {
    /// ```math
    /// (\frac{\partial E}{\partial h_i} \cdot \frac{\partial h_i}{\partial f})
    /// ```
    ///
    /// - $h_i$: the $i$-th immediate successor of this node
    /// - $f$: this node
    pub addends_of_gradient_of_root_at_this: Vec<f64>,

    pub evaluate_cache: EvaluateCache,
}
