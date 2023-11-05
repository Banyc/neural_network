use std::sync::{Arc, Mutex};

use rand::Rng;

use super::nodes::node::{
    do_gradient_descent_steps_and_reset_caches_on_all_nodes,
    do_gradient_descent_steps_on_all_nodes, reset_caches_on_all_nodes, GeneralNode,
};

pub struct NeuralNetwork {
    terminal_node: Arc<Mutex<GeneralNode>>,
    error_node: Arc<Mutex<GeneralNode>>,
    label_index: usize,
    step_size: f64,
}

impl NeuralNetwork {
    fn check_rep(&self) {}

    pub fn new(
        terminal_node: Arc<Mutex<GeneralNode>>,
        error_node: Arc<Mutex<GeneralNode>>,
        label_index: usize,
        step_size: f64,
    ) -> NeuralNetwork {
        let this = NeuralNetwork {
            terminal_node,
            error_node,
            label_index,
            step_size,
        };
        this.check_rep();
        this
    }

    pub fn backpropagation_step_and_reset_caches(&self, inputs: &[f64]) {
        self.compute_error(inputs);
        do_gradient_descent_steps_and_reset_caches_on_all_nodes(&self.error_node, self.step_size);
    }

    pub fn backpropagation_step(&self, inputs: &[f64]) {
        self.compute_error(inputs);
        do_gradient_descent_steps_on_all_nodes(&self.error_node, self.step_size);
    }

    pub fn evaluate_and_reset_caches(&self, inputs: &[f64]) -> f64 {
        let ret = self.evaluate(inputs);
        reset_caches_on_all_nodes(&self.terminal_node);
        ret
    }

    pub fn evaluate(&self, inputs: &[f64]) -> f64 {
        let mut terminal_node = self.terminal_node.lock().unwrap();
        terminal_node.evaluate(inputs)
    }

    pub fn compute_error_and_reset_caches(&self, inputs: &[f64]) -> f64 {
        let ret = self.compute_error(inputs);
        reset_caches_on_all_nodes(&self.error_node);
        ret
    }

    pub fn compute_error(&self, inputs: &[f64]) -> f64 {
        let mut error_node = self.error_node.lock().unwrap();
        error_node.evaluate(inputs)
    }

    pub fn train<S>(&self, dataset: &[S], max_steps: usize)
    where
        S: AsRef<[f64]>,
    {
        let mut rng = rand::thread_rng();
        for i in 0..max_steps {
            let dataset_index: usize = rng.gen_range(0..dataset.len());
            self.backpropagation_step_and_reset_caches(dataset[dataset_index].as_ref());
            if i % (max_steps / 10) == 0 {
                println!("{:.2}%", (100 * i) as f64 / max_steps as f64);
            }
        }
    }

    pub fn errors_on_dataset<S>(&self, dataset: &[S]) -> f64
    where
        S: AsRef<[f64]>,
    {
        let mut errors = 0;
        for inputs in dataset {
            let eval = self.evaluate_and_reset_caches(inputs.as_ref());
            if (eval - inputs.as_ref()[self.label_index]).abs() >= 0.5 {
                errors += 1;
            }
        }
        errors as f64 / dataset.len() as f64
    }
}
