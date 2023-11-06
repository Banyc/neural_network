//! # Terminologies
//!
//! - $f$: the function represented by the node
//! - $z$: the functions represented by the operands (predecessors) of the node
//!   - outputs of those functions become the input of the node
//! - $w$: the tunable parameters of $f$
//! - $E$: the outmost function represented by the root node of the computation graph

use std::{cell::RefCell, collections::VecDeque, rc::Rc};

use thiserror::Error;

use super::utils::cached_node_data::CachedNodeData;

pub trait NodeComputation {
    fn compute_output(&self, parameters: &[f64], operand_outputs: &[f64], inputs: &[f64]) -> f64;

    /// $$
    /// \frac{\partial f}{\partial z}
    /// $$
    ///
    /// - $z$: the non-tunable operands of $f$
    fn compute_gradient_of_this_at_operand(
        &self,
        parameters: &[f64],
        operand_outputs: &[f64],
    ) -> Vec<f64>;

    /// $$
    /// \frac{\partial f}{\partial w}
    /// $$
    ///
    /// - $w$: the tunable parameters of $f$
    fn compute_gradient_of_this_at_parameter(
        &self,
        parameters: &[f64],
        operand_outputs: &[f64],
    ) -> Vec<f64>;
}

pub struct GeneralNode {
    parameters: Vec<f64>,
    operands: Vec<Rc<RefCell<GeneralNode>>>,
    successor_len: usize,
    cache: CachedNodeData,
    computation: Box<dyn NodeComputation + Sync + Send>,
}

impl GeneralNode {
    fn check_rep(&self) {
        if let Some(gradient) = &self.cache.gradient_of_root_at_parameter {
            assert_eq!(gradient.len(), self.parameters.len());
        }
        if let Some(gradient) = &self.cache.gradient_of_function_at_parameter {
            assert_eq!(gradient.len(), self.parameters.len());
        }
        if let Some(gradient) = &self.cache.gradient_of_function_at_operand {
            assert_eq!(gradient.len(), self.operands.len());
        }
        assert_eq!(
            self.cache.output.is_none(),
            self.cache.operand_outputs.is_none()
        );
        assert!(self.cache.addends_of_gradient_of_root_at_function.len() <= self.successor_len);
        if self.cache.gradient_of_root_at_function.is_some() {
            assert_eq!(
                self.cache.addends_of_gradient_of_root_at_function.len(),
                self.successor_len
            );
        }
    }

    pub fn new(
        operands: Vec<Rc<RefCell<GeneralNode>>>,
        computation: Box<dyn NodeComputation + Sync + Send>,
        parameters: Vec<f64>,
    ) -> GeneralNode {
        operands.iter().for_each(|operand| {
            let mut operand = operand.borrow_mut();
            operand.increment_successor_len();
        });
        let this = Self {
            parameters,
            operands,
            successor_len: 0,
            cache: CachedNodeData::new(),
            computation,
        };
        this.check_rep();
        this
    }

    /// The output is cached until reset
    pub fn evaluate(&mut self, inputs: &[f64]) -> f64 {
        self.cache.output.get_or_insert_with(|| {
            assert!(self.cache.operand_outputs.is_none());
            let operand_outputs: Rc<_> = self
                .operands
                .iter_mut()
                .map(|operand| {
                    let mut operand = operand.borrow_mut();
                    operand.evaluate(inputs)
                })
                .collect();
            let ret = self
                .computation
                .compute_output(&self.parameters, &operand_outputs, inputs);
            self.cache.operand_outputs = Some(operand_outputs);
            ret
        });
        self.check_rep();
        assert!(self.cache.operand_outputs.is_some());
        self.cache.output.unwrap()
    }

    pub fn do_gradient_descent_step_and_reset_cache(
        &mut self,
        step_size: f64,
    ) -> Result<(), GradientDescentError> {
        self.do_gradient_descent_step(step_size)?;
        self.cache.reset();
        self.check_rep();
        Ok(())
    }

    pub fn do_gradient_descent_step(&mut self, step_size: f64) -> Result<(), GradientDescentError> {
        if self.successor_len > self.cache.addends_of_gradient_of_root_at_function.len() {
            return Err(GradientDescentError::NotReceivingEnoughAddendsOfGradientFromSuccessors);
        }
        if self.cache.output.is_none() || self.cache.operand_outputs.is_none() {
            return Err(GradientDescentError::NoEvaluationOutputCaches);
        }
        assert_eq!(
            self.successor_len,
            self.cache.addends_of_gradient_of_root_at_function.len()
        );
        self.distribute_global_gradient_addends_to_operands();
        self.adjust_parameters(step_size);
        self.check_rep();
        Ok(())
    }

    fn increment_successor_len(&mut self) {
        self.successor_len += 1;
        self.check_rep();
    }

    fn adjust_parameters(&mut self, step_size: f64) {
        let gradient = Rc::clone(self.gradient_of_root_at_parameter().unwrap());
        gradient.iter().enumerate().for_each(|(i, gradient_entry)| {
            self.parameters[i] -= step_size * *gradient_entry;
        });
        self.check_rep();
    }

    fn distribute_global_gradient_addends_to_operands(&mut self) {
        let gradient_of_this_at_operand = Rc::clone(self.gradient_of_this_at_operand().unwrap());
        let gradient_of_root_at_this = self.gradient_of_root_at_this().unwrap();
        if self
            .cache
            .has_distributed_addend_of_gradient_of_root_at_predecessor
        {
            panic!();
        }
        self.cache
            .has_distributed_addend_of_gradient_of_root_at_predecessor = true;
        (0..self.operands.len()).for_each(|i| {
            // $$
            // \frac{\partial E}{\partial f} \cdot \frac{\partial f}{\partial z}
            // $$
            let addend_of_gradient_of_root_at_predecessor =
                gradient_of_root_at_this * gradient_of_this_at_operand[i];
            let mut operand = self.operands[i].borrow_mut();
            operand
                .add_addend_of_gradient_of_root_at_this(addend_of_gradient_of_root_at_predecessor);
        });
        self.check_rep();
    }

    fn add_addend_of_gradient_of_root_at_this(&mut self, gradient_addend: f64) {
        assert!(self.cache.gradient_of_root_at_function.is_none());
        self.cache
            .addends_of_gradient_of_root_at_function
            .push(gradient_addend);
        self.check_rep();
    }

    /// $$
    /// \frac{\partial f}{\partial z}
    /// $$
    ///
    /// - $z$: the non-tunable operands of $f$
    pub fn gradient_of_this_at_operand(
        &mut self,
    ) -> Result<&Rc<[f64]>, GradientOfThisAtOperandError> {
        let operand_outputs = self
            .operand_outputs()
            .ok_or(GradientOfThisAtOperandError::NoEvaluationOutputCaches)?;
        if self.cache.gradient_of_function_at_operand.is_none() {
            self.cache.gradient_of_function_at_operand = Some(
                self.computation
                    .compute_gradient_of_this_at_operand(&self.parameters, operand_outputs)
                    .into(),
            );
        }
        self.check_rep();
        Ok(self.cache.gradient_of_function_at_operand.as_ref().unwrap())
    }

    /// $$
    /// \frac{\partial E}{\partial f}
    /// $$
    ///
    /// - $E$: the out-most function of the entire network
    pub fn gradient_of_root_at_this(&mut self) -> Result<f64, GradientOfRootAtThisError> {
        if self.successor_len != self.cache.addends_of_gradient_of_root_at_function.len() {
            return Err(
                GradientOfRootAtThisError::NotReceivingEnoughAddendsOfGradientFromSuccessors,
            );
        }
        self.cache
            .gradient_of_root_at_function
            .get_or_insert_with(|| {
                assert_eq!(
                    self.successor_len,
                    self.cache.addends_of_gradient_of_root_at_function.len()
                );
                if self.successor_len == 0 {
                    // this is the root node
                    1.0
                } else {
                    self.cache
                        .addends_of_gradient_of_root_at_function
                        .iter()
                        .sum()
                }
            });
        self.check_rep();
        Ok(self.cache.gradient_of_root_at_function.unwrap())
    }

    /// $$
    /// \frac{\partial f}{\partial w}
    /// $$
    ///
    /// - $w$: the tunable parameters of $f$
    pub fn gradient_of_this_at_parameter(
        &mut self,
    ) -> Result<&Rc<[f64]>, GradientOfThisAtParameterError> {
        let operand_outputs = self
            .operand_outputs()
            .ok_or(GradientOfThisAtParameterError::NoEvaluationOutputCaches)?;
        if self.cache.gradient_of_function_at_parameter.is_none() {
            self.cache.gradient_of_function_at_parameter = Some(
                self.computation
                    .compute_gradient_of_this_at_parameter(
                        &self.parameters,
                        operand_outputs.as_ref(),
                    )
                    .into(),
            );
        }
        self.check_rep();
        Ok(self
            .cache
            .gradient_of_function_at_parameter
            .as_ref()
            .unwrap())
    }

    /// $$
    /// \frac{\partial E}{\partial w}
    /// $$
    ///
    /// - $w$: the tunable parameters of $f$
    pub fn gradient_of_root_at_parameter(
        &mut self,
    ) -> Result<&Rc<[f64]>, GradientOfRootAtParameterError> {
        let gradient_of_this_at_parameter = Rc::clone(
            self.gradient_of_this_at_parameter()
                .map_err(GradientOfRootAtParameterError::GradientOfThisAtParameter)?,
        );
        let gradient_of_root_at_this = self
            .gradient_of_root_at_this()
            .map_err(GradientOfRootAtParameterError::GradientOfRootAtThis)?;
        self.cache
            .gradient_of_root_at_parameter
            .get_or_insert_with(|| {
                gradient_of_this_at_parameter
                    .iter()
                    .map(|partial_derivative_of_this_at_parameter_i| {
                        gradient_of_root_at_this * *partial_derivative_of_this_at_parameter_i
                    })
                    .collect()
            });
        self.check_rep();
        Ok(self.cache.gradient_of_root_at_parameter.as_ref().unwrap())
    }

    pub fn operand_outputs(&self) -> Option<&Rc<[f64]>> {
        self.cache.operand_outputs.as_ref()
    }

    pub fn output(&self) -> Option<f64> {
        self.cache.output
    }

    pub fn parameters(&self) -> &Vec<f64> {
        &self.parameters
    }
}

pub fn clone_node_batch(nodes: &[Rc<RefCell<GeneralNode>>]) -> Vec<Rc<RefCell<GeneralNode>>> {
    nodes.iter().map(Rc::clone).collect()
}

/// Inefficient: same node might be visited more than once
pub fn reset_caches_on_all_nodes(root_note: &Rc<RefCell<GeneralNode>>) {
    let f = |n: &mut GeneralNode| {
        n.cache.reset();
    };
    bfs_operands(root_note, f);
}

pub fn do_gradient_descent_steps_and_reset_caches_on_all_nodes(
    root_note: &Rc<RefCell<GeneralNode>>,
    step_size: f64,
) {
    let f = |n: &mut GeneralNode| {
        match n.do_gradient_descent_step_and_reset_cache(step_size) {
            Ok(_) => (),
            Err(e) => match e {
                GradientDescentError::NotReceivingEnoughAddendsOfGradientFromSuccessors => (),
                // haven't evaluate before gradient descent
                GradientDescentError::NoEvaluationOutputCaches => panic!(),
            },
        };
    };
    bfs_operands(root_note, f);
}

pub fn do_gradient_descent_steps_on_all_nodes(
    root_note: &Rc<RefCell<GeneralNode>>,
    step_size: f64,
) {
    let f = |n: &mut GeneralNode| {
        match n.do_gradient_descent_step(step_size) {
            Ok(_) => (),
            Err(e) => match e {
                GradientDescentError::NotReceivingEnoughAddendsOfGradientFromSuccessors => (),
                // haven't evaluate before gradient descent
                GradientDescentError::NoEvaluationOutputCaches => panic!(),
            },
        };
    };
    bfs_operands(root_note, f);
}

fn bfs_operands(root_node: &Rc<RefCell<GeneralNode>>, f: impl Fn(&mut GeneralNode)) {
    let mut q = VecDeque::new();
    q.push_back(Rc::clone(root_node));

    while let Some(n) = q.pop_front() {
        let mut n = n.borrow_mut();
        f(&mut n);
        for op in &n.operands {
            q.push_back(Rc::clone(op));
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
