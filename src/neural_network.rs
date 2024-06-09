use std::{
    sync::{Arc, Mutex},
    time::{Duration, Instant},
};

use rand::Rng;

use crate::node::graph_delete_caches;

use super::node::{graph_do_gradient_descent_steps, Node};

#[derive(Debug)]
pub struct NeuralNetwork {
    /// output: a prediction
    terminal_nodes: Vec<Arc<Mutex<Node>>>,
    /// output: the error between the prediction and the label
    error_node: Arc<Mutex<Node>>,
    /// the index of the input node which accepts a label
    label_indices: Vec<usize>,
    /// learning rate
    step_size: f64,
}
impl NeuralNetwork {
    fn check_rep(&self) {}

    pub fn new(
        terminal_nodes: Vec<Arc<Mutex<Node>>>,
        error_node: Arc<Mutex<Node>>,
        label_indices: Vec<usize>,
        step_size: f64,
    ) -> NeuralNetwork {
        assert_eq!(terminal_nodes.len(), label_indices.len());
        let this = NeuralNetwork {
            terminal_nodes,
            error_node,
            label_indices,
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

    pub fn evaluate(&mut self, inputs: &[f64]) -> Vec<f64> {
        let mut outputs = vec![];

        for terminal_node in &self.terminal_nodes {
            let mut terminal_node = terminal_node.lock().unwrap();
            let batch_index = 0;
            let output = terminal_node.evaluate_once(inputs, batch_index);
            outputs.push(output);
        }
        for terminal_node in &self.terminal_nodes {
            graph_delete_caches(terminal_node);
        }
        self.check_rep();
        outputs
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
        let mut progress_printer = ProgressPrinter::new();
        for i in 0..max_steps {
            batch_input.clear();
            for _ in 0..batch_size {
                let dataset_index: usize = rng.gen_range(0..dataset.len());
                batch_input.push(dataset[dataset_index].as_ref());
            }
            self.backpropagation_step(&batch_input);
            progress_printer.print_progress(i, max_steps);
        }
        self.check_rep();
    }

    pub fn accuracy<S>(
        &mut self,
        dataset: &[S],
        accurate: impl Fn(Vec<f64>, Vec<f64>) -> bool,
    ) -> f64
    where
        S: AsRef<[f64]>,
    {
        let mut progress_printer = ProgressPrinter::new();
        let mut accurate_count = 0;
        for (i, inputs) in dataset.iter().enumerate() {
            let eval = self.evaluate(inputs.as_ref());
            let label = self
                .label_indices
                .iter()
                .copied()
                .map(|i| inputs.as_ref()[i])
                .collect::<Vec<f64>>();
            if accurate(eval, label) {
                accurate_count += 1;
            }
            progress_printer.print_progress(i, dataset.len());
        }
        self.check_rep();
        accurate_count as f64 / dataset.len() as f64
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

struct ProgressPrinter {
    now: Instant,
}
impl ProgressPrinter {
    pub fn new() -> Self {
        Self {
            now: Instant::now(),
        }
    }

    fn print_progress(&mut self, i: usize, len: usize) {
        let gap = len / 10;
        if gap != 0 && i % gap != 0 {
            return;
        }
        let percentage = (100 * i) as f64 / len as f64;
        let elapsed = human_duration(self.now.elapsed());
        self.now = Instant::now();
        println!("{percentage:.2}%; {elapsed}");
    }
}

fn human_duration(duration: Duration) -> String {
    let seconds = duration.as_secs_f64();
    let minutes = seconds / 60.;
    let hours = minutes / 60.;
    let milliseconds = seconds * 1_000.;
    let microseconds = milliseconds * 1_000.;
    let nanoseconds = microseconds * 1_000.;
    if 1. < hours {
        return format!("{hours:.2} h");
    }
    if 1. < minutes {
        return format!("{minutes:.2} min");
    }
    if 1. < seconds {
        return format!("{seconds:.2} s");
    }
    if 1. < milliseconds {
        return format!("{milliseconds:.2} ms");
    }
    if 1. < microseconds {
        return format!("{microseconds:.2} us");
    }
    format!("{nanoseconds:.2} ns")
}
