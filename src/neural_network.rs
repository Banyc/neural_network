use std::sync::{Arc, Mutex};

use rand::Rng;

use super::nodes::node::{do_gradient_descent_step_on_all_nodes, GeneralNode};

pub struct NeuralNetwork {
    terminal_node: Arc<Mutex<GeneralNode>>,
    error_node: Arc<Mutex<GeneralNode>>,
    step_size: f64,
}

impl NeuralNetwork {
    pub fn backpropagation_step(&self, inputs: &Vec<f64>) {
        {
            let mut error_node = self.error_node.lock().unwrap();
            error_node.evaluate(inputs);
        }
        do_gradient_descent_step_on_all_nodes(&self.error_node, self.step_size);
    }

    pub fn evaluate(&self, inputs: &Vec<f64>) -> f64 {
        let mut terminal_node = self.terminal_node.lock().unwrap();
        terminal_node.evaluate(inputs)
    }

    pub fn train(&self, dataset: Vec<InputOutputPair>, max_steps: usize) {
        let mut rng = rand::thread_rng();
        for i in 0..max_steps {
            let dataset_index: usize = rng.gen_range(0..dataset.len());
            self.backpropagation_step(&dataset[dataset_index].inputs);
            if i % (max_steps / 10) == 0 {
                println!("{:.2}%", (100 * i) as f64 / max_steps as f64);
            }
        }
    }

    pub fn errors_on_dataset(&self, dataset: Vec<InputOutputPair>) -> f64 {
        let mut errors = 0;
        for pair in &dataset {
            let eval = self.evaluate(&pair.inputs);
            if eval >= 0.5 && pair.label() >= 0.5 || eval < 0.5 && pair.label() < 0.5 {
            } else {
                errors += 1;
            }
        }
        errors as f64 / dataset.len() as f64
    }
}

pub struct InputOutputPair {
    pub inputs: Vec<f64>,
    pub label_index: usize,
}

impl InputOutputPair {
    pub fn label(&self) -> f64 {
        self.inputs[self.label_index]
    }
}
