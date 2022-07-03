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

    /// Every time overwrite cache
    pub fn evaluate(&mut self, inputs: &Vec<f64>) -> f64 {
        let mut operand_outputs = Vec::new();
        for operand in self.operands.iter_mut() {
            let mut operand = operand.lock().unwrap();
            operand_outputs.push(operand.evaluate(&inputs));
        }
        let output = self
            .computation
            .compute_output(&self.parameters, &operand_outputs, inputs);
        self.cache.operand_outputs = Some(Arc::new(operand_outputs));
        self.cache.output = Some(output);
        self.check_rep();
        output
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
        Ok(())
    }

    fn increment_successor_len(&mut self) {
        self.successor_len += 1;
    }

    fn adjust_parameters(&mut self, step_size: f64) {
        let gradient = self.global_parameter_gradient();
        for (i, gradient_entry) in gradient.iter().enumerate() {
            self.parameters[i] -= step_size * *gradient_entry;
        }
        self.check_rep();
    }

    fn distribute_global_gradient_entries_to_operands(&mut self) {
        if self.cache.has_distributed_global_gradient_entries {
            panic!();
        }
        self.cache.has_distributed_global_gradient_entries = true;
        for i in 0..self.operands.len() {
            let gradient_entry = self.global_gradient() * self.local_operand_gradient()[i];
            let mut operand = self.operands[i].lock().unwrap();
            operand.add_global_gradient_entry(gradient_entry);
        }
        self.check_rep();
    }

    fn add_global_gradient_entry(&mut self, gradient_entry: f64) {
        assert!(self.cache.global_gradient.is_none());
        self.cache.global_gradient_entries.push(gradient_entry);
    }

    /// $$
    /// \frac{\partial f}{\partial z}
    /// $$
    ///
    /// - $z$: the non-tunable operands of $f$
    fn local_operand_gradient(&mut self) -> Arc<Vec<f64>> {
        // TODO: throw error if None
        let operand_outputs = self.operand_outputs().unwrap();
        self.cache.local_operand_gradient.get_or_insert_with(|| {
            Arc::new(
                self.computation
                    .compute_local_operand_gradient(&self.parameters, operand_outputs.as_ref()),
            )
        });
        self.check_rep();
        Arc::clone(self.cache.local_operand_gradient.as_ref().unwrap())
    }

    /// $$
    /// \frac{\partial E}{\partial f}
    /// $$
    ///
    /// - $E$: the out-most function of the entire network
    fn global_gradient(&mut self) -> f64 {
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
        self.cache.global_gradient.unwrap()
    }

    /// $$
    /// \frac{\partial f}{\partial w}
    /// $$
    ///
    /// - $w$: the tunable parameters of $f$
    fn local_parameter_gradient(&mut self) -> Arc<Vec<f64>> {
        // TODO: throw error if None
        let operand_outputs = self.operand_outputs().unwrap();
        self.cache.local_parameter_gradient.get_or_insert_with(|| {
            Arc::new(
                self.computation
                    .compute_local_parameter_gradient(&self.parameters, operand_outputs.as_ref()),
            )
        });
        self.check_rep();
        Arc::clone(self.cache.local_parameter_gradient.as_ref().unwrap())
    }

    /// $$
    /// \frac{\partial E}{\partial w}
    /// $$
    ///
    /// - $w$: the tunable parameters of $f$
    fn global_parameter_gradient(&mut self) -> Arc<Vec<f64>> {
        let local_parameter_gradient = self.local_parameter_gradient();
        let global_gradient = self.global_gradient();
        self.cache.global_parameter_gradient.get_or_insert_with(|| {
            let mut gradient_entries = Vec::new();
            for local_parameter_gradient_entry in local_parameter_gradient.iter() {
                gradient_entries.push(global_gradient * *local_parameter_gradient_entry);
            }
            Arc::new(gradient_entries)
        });
        self.check_rep();
        Arc::clone(self.cache.global_parameter_gradient.as_ref().unwrap())
    }

    fn operand_outputs(&self) -> Option<Arc<Vec<f64>>> {
        match &self.cache.operand_outputs {
            Some(x) => Some(Arc::clone(&x)),
            None => None,
        }
    }
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
