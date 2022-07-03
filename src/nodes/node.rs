use std::{
    collections::VecDeque,
    sync::{Arc, Mutex},
};

use super::utils::cached_node_data::CachedNodeData;

pub trait NodeComputation {
    fn compute_output(
        &self,
        parameters: &Vec<f64>,
        operand_outputs: &Vec<f64>,
        inputs: &Vec<f64>,
    ) -> f64;
    fn compute_local_operand_gradient(
        &self,
        parameters: &Vec<f64>,
        operand_outputs: &Vec<f64>,
    ) -> Vec<f64>;
    fn compute_local_parameter_gradient(
        &self,
        parameters: &Vec<f64>,
        operand_outputs: &Vec<f64>,
    ) -> Vec<f64>;
}

pub struct GeneralNode {
    parameters: Vec<f64>,
    operands: Vec<Arc<Mutex<GeneralNode>>>,
    successor_len: usize,
    cache: CachedNodeData,
    computation: Box<dyn NodeComputation>,
}

impl GeneralNode {
    fn check_rep(&self) {
        if let Some(gradient) = &self.cache.global_parameter_gradient {
            assert_eq!(gradient.len(), self.parameters.len());
        }
        if let Some(gradient) = &self.cache.local_parameter_gradient {
            assert_eq!(gradient.len(), self.parameters.len());
        }
        if let Some(gradient) = &self.cache.local_operand_gradient {
            assert_eq!(gradient.len(), self.operands.len());
        }
        assert_eq!(
            self.cache.output.is_none(),
            self.cache.operand_outputs.is_none()
        );
        assert!(self.cache.global_gradient_entries.len() <= self.successor_len);
        if self.cache.global_gradient.is_some() {
            assert_eq!(self.cache.global_gradient_entries.len(), self.successor_len);
        }
    }

    pub fn new(
        operands: Vec<Arc<Mutex<GeneralNode>>>,
        computation: Box<dyn NodeComputation>,
        parameters: Vec<f64>,
    ) -> GeneralNode {
        for operand in &operands {
            let mut operand = operand.lock().unwrap();
            operand.increment_successor_len();
        }
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
    pub fn evaluate(&mut self, inputs: &Vec<f64>) -> f64 {
        self.cache.output.get_or_insert_with(|| {
            assert!(self.cache.operand_outputs.is_none());
            let mut operand_outputs = Vec::new();
            for operand in self.operands.iter_mut() {
                let mut operand = operand.lock().unwrap();
                operand_outputs.push(operand.evaluate(&inputs));
            }
            let ret = self
                .computation
                .compute_output(&self.parameters, &operand_outputs, inputs);
            self.cache.operand_outputs = Some(Arc::new(operand_outputs));
            ret
        });
        self.check_rep();
        assert!(self.cache.operand_outputs.is_some());
        self.cache.output.unwrap()
    }

    pub fn do_gradient_descent_step(&mut self, step_size: f64) -> Result<(), GradientDescentError> {
        if self.successor_len > self.cache.global_gradient_entries.len() {
            return Err(
                GradientDescentError::NotReceivingEnoughGlobalGradientEntriesFromSuccessors,
            );
        }
        if self.cache.output.is_none() || self.cache.operand_outputs.is_none() {
            return Err(GradientDescentError::NoEvaluationOutputCaches);
        }
        assert_eq!(self.successor_len, self.cache.global_gradient_entries.len());
        self.distribute_global_gradient_entries_to_operands();
        self.adjust_parameters(step_size);
        self.cache.reset();
        self.check_rep();
        Ok(())
    }

    fn increment_successor_len(&mut self) {
        self.successor_len += 1;
        self.check_rep();
    }

    fn adjust_parameters(&mut self, step_size: f64) {
        let gradient = self.global_parameter_gradient().unwrap();
        for (i, gradient_entry) in gradient.iter().enumerate() {
            self.parameters[i] -= step_size * *gradient_entry;
        }
        self.check_rep();
    }

    fn distribute_global_gradient_entries_to_operands(&mut self) {
        let local_operand_gradient = self.local_operand_gradient().unwrap();
        let global_gradient = self.global_gradient().unwrap();
        if self.cache.has_distributed_global_gradient_entries {
            panic!();
        }
        self.cache.has_distributed_global_gradient_entries = true;
        for i in 0..self.operands.len() {
            let gradient_entry = global_gradient * local_operand_gradient[i];
            let mut operand = self.operands[i].lock().unwrap();
            operand.add_global_gradient_entry(gradient_entry);
        }
        self.check_rep();
    }

    fn add_global_gradient_entry(&mut self, gradient_entry: f64) {
        assert!(self.cache.global_gradient.is_none());
        self.cache.global_gradient_entries.push(gradient_entry);
        self.check_rep();
    }

    /// $$
    /// \frac{\partial f}{\partial z}
    /// $$
    ///
    /// - $z$: the non-tunable operands of $f$
    pub fn local_operand_gradient(&mut self) -> Result<Arc<Vec<f64>>, LocalOperandGradientError> {
        let operand_outputs = self
            .operand_outputs()
            .ok_or(LocalOperandGradientError::NoEvaluationOutputCaches)?;
        self.cache.local_operand_gradient.get_or_insert_with(|| {
            Arc::new(
                self.computation
                    .compute_local_operand_gradient(&self.parameters, operand_outputs.as_ref()),
            )
        });
        self.check_rep();
        Ok(Arc::clone(
            self.cache.local_operand_gradient.as_ref().unwrap(),
        ))
    }

    /// $$
    /// \frac{\partial E}{\partial f}
    /// $$
    ///
    /// - $E$: the out-most function of the entire network
    pub fn global_gradient(&mut self) -> Result<f64, GlobalGradientError> {
        if self.successor_len != self.cache.global_gradient_entries.len() {
            return Err(GlobalGradientError::NotReceivingEnoughGlobalGradientEntriesFromSuccessors);
        }
        self.cache.global_gradient.get_or_insert_with(|| {
            assert_eq!(self.successor_len, self.cache.global_gradient_entries.len());
            if self.successor_len == 0 {
                // this is the root node
                1.0
            } else {
                self.cache.global_gradient_entries.iter().sum()
            }
        });
        self.check_rep();
        Ok(self.cache.global_gradient.unwrap())
    }

    /// $$
    /// \frac{\partial f}{\partial w}
    /// $$
    ///
    /// - $w$: the tunable parameters of $f$
    pub fn local_parameter_gradient(
        &mut self,
    ) -> Result<Arc<Vec<f64>>, LocalParameterGradientError> {
        let operand_outputs = self
            .operand_outputs()
            .ok_or(LocalParameterGradientError::NoEvaluationOutputCaches)?;
        self.cache.local_parameter_gradient.get_or_insert_with(|| {
            Arc::new(
                self.computation
                    .compute_local_parameter_gradient(&self.parameters, operand_outputs.as_ref()),
            )
        });
        self.check_rep();
        Ok(Arc::clone(
            self.cache.local_parameter_gradient.as_ref().unwrap(),
        ))
    }

    /// $$
    /// \frac{\partial E}{\partial w}
    /// $$
    ///
    /// - $w$: the tunable parameters of $f$
    pub fn global_parameter_gradient(
        &mut self,
    ) -> Result<Arc<Vec<f64>>, GlobalParameterGradientError> {
        let local_parameter_gradient = self
            .local_parameter_gradient()
            .map_err(|e| GlobalParameterGradientError::LocalParameterGradientError(e))?;
        let global_gradient = self
            .global_gradient()
            .map_err(|e| GlobalParameterGradientError::GlobalGradientError(e))?;
        self.cache.global_parameter_gradient.get_or_insert_with(|| {
            let mut gradient_entries = Vec::new();
            for local_parameter_gradient_entry in local_parameter_gradient.iter() {
                gradient_entries.push(global_gradient * *local_parameter_gradient_entry);
            }
            Arc::new(gradient_entries)
        });
        self.check_rep();
        Ok(Arc::clone(
            self.cache.global_parameter_gradient.as_ref().unwrap(),
        ))
    }

    pub fn operand_outputs(&self) -> Option<Arc<Vec<f64>>> {
        match &self.cache.operand_outputs {
            Some(x) => Some(Arc::clone(&x)),
            None => None,
        }
    }
}

pub fn clone_node_batch(nodes: &Vec<Arc<Mutex<GeneralNode>>>) -> Vec<Arc<Mutex<GeneralNode>>> {
    let mut cloned_nodes = Vec::new();
    for node in nodes {
        cloned_nodes.push(Arc::clone(node));
    }
    cloned_nodes
}

pub fn do_gradient_descent_step_on_all_nodes(root_note: &Arc<Mutex<GeneralNode>>, step_size: f64) {
    let f = |n: &mut GeneralNode| {
        match n.do_gradient_descent_step(step_size) {
            Ok(_) => (),
            Err(e) => match e {
                GradientDescentError::NotReceivingEnoughGlobalGradientEntriesFromSuccessors => (),
                // haven't evaluate before gradient descent
                GradientDescentError::NoEvaluationOutputCaches => panic!(),
            },
        };
    };
    bfs_operands(root_note, f);
}

fn bfs_operands(root_node: &Arc<Mutex<GeneralNode>>, f: impl Fn(&mut GeneralNode) -> ()) {
    let mut q = VecDeque::new();
    q.push_back(Arc::clone(root_node));

    while let Some(n) = q.pop_front() {
        let mut n = n.lock().unwrap();
        f(&mut n);
        for op in &n.operands {
            q.push_back(Arc::clone(op));
        }
    }
}

pub enum GradientDescentError {
    NotReceivingEnoughGlobalGradientEntriesFromSuccessors,
    NoEvaluationOutputCaches,
}

#[derive(Debug)]
pub enum GlobalGradientError {
    NotReceivingEnoughGlobalGradientEntriesFromSuccessors,
}

#[derive(Debug)]
pub enum GlobalParameterGradientError {
    GlobalGradientError(GlobalGradientError),
    LocalParameterGradientError(LocalParameterGradientError),
}

#[derive(Debug)]
pub enum LocalParameterGradientError {
    NoEvaluationOutputCaches,
}

#[derive(Debug)]
pub enum LocalOperandGradientError {
    NoEvaluationOutputCaches,
}
