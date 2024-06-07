use std::sync::{Arc, Mutex};

use rand::Rng;

use crate::node::graph_delete_caches;

use super::node::{graph_do_gradient_descent_steps, Node};

#[derive(Debug)]
pub struct NeuralNetwork {
    /// output: a prediction
    terminal_node: Arc<Mutex<Node>>,
    /// output: the error between the prediction and the label
    error_node: Arc<Mutex<Node>>,
    /// the index of the input node which accepts a label
    label_index: usize,
    /// learning rate
    step_size: f64,
}
impl NeuralNetwork {
    fn check_rep(&self) {}

    pub fn new(
        terminal_node: Arc<Mutex<Node>>,
        error_node: Arc<Mutex<Node>>,
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

    pub fn backpropagation_step<I>(&mut self, samples: &[I])
    where
        I: AsRef<[f64]>,
    {
        for (batch_index, inputs) in samples.iter().enumerate() {
            let inputs = inputs.as_ref();
            self.compute_error(inputs, EvalOption::KeepCache, batch_index);
        }
        graph_do_gradient_descent_steps(&self.error_node, self.step_size);
        self.check_rep();
    }

    pub fn evaluate(&mut self, inputs: &[f64]) -> f64 {
        let mut terminal_node = self.terminal_node.lock().unwrap();
        let batch_index = 0;
        let output = terminal_node.evaluate_once(inputs, batch_index);
        drop(terminal_node);
        graph_delete_caches(&self.terminal_node);
        self.check_rep();
        output
    }

    pub fn compute_error(&mut self, inputs: &[f64], option: EvalOption, batch_index: usize) -> f64 {
        let mut error_node = self.error_node.lock().unwrap();
        let output = error_node.evaluate_once(inputs, batch_index);
        drop(error_node);
        if matches!(option, EvalOption::ClearCache) {
            graph_delete_caches(&self.error_node);
        }
        self.check_rep();
        output
    }

    pub fn train<S>(&mut self, dataset: &[S], max_steps: usize, option: TrainOption)
    where
        S: AsRef<[f64]>,
    {
        let batch_size = match option {
            TrainOption::StochasticGradientDescent => 1,
            TrainOption::MiniBatchGradientDescent { batch_size } => batch_size,
            TrainOption::BatchGradientDescent => dataset.len(),
        };
        let mut batch_input = vec![];
        let mut rng = rand::thread_rng();
        for i in 0..max_steps {
            batch_input.clear();
            for _ in 0..batch_size {
                let dataset_index: usize = rng.gen_range(0..dataset.len());
                batch_input.push(dataset[dataset_index].as_ref());
            }
            self.backpropagation_step(&batch_input);
            if i % (max_steps / 10) == 0 {
                println!("{:.2}%", (100 * i) as f64 / max_steps as f64);
            }
        }
        self.check_rep();
    }

    pub fn errors_on_dataset<S>(&mut self, dataset: &[S]) -> f64
    where
        S: AsRef<[f64]>,
    {
        let mut errors = 0;
        for inputs in dataset {
            let eval = self.evaluate(inputs.as_ref());
            if (eval - inputs.as_ref()[self.label_index]).abs() >= 0.5 {
                errors += 1;
            }
        }
        self.check_rep();
        errors as f64 / dataset.len() as f64
    }
}

#[derive(Debug, Clone, Copy)]
pub enum EvalOption {
    KeepCache,
    ClearCache,
}

#[derive(Debug, Clone, Copy)]
pub enum TrainOption {
    StochasticGradientDescent,
    MiniBatchGradientDescent { batch_size: usize },
    BatchGradientDescent,
}
