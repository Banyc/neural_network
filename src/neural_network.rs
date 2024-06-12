use std::time::{Duration, Instant};

use rand::Rng;

use crate::node::{graph_delete_caches, NodeContext, SharedNode};

use super::node::graph_do_gradient_descent_steps;

#[derive(Debug)]
pub struct NeuralNetwork {
    /// output: a prediction
    terminal_nodes: Vec<SharedNode>,
    /// output: the error between the prediction and the label
    error_node: SharedNode,
    /// global necessity
    cx: NodeContext,
}
impl NeuralNetwork {
    fn check_rep(&self) {}

    pub fn new(terminal_nodes: Vec<SharedNode>, error_node: SharedNode) -> NeuralNetwork {
        let cx = NodeContext::new();
        let this = NeuralNetwork {
            terminal_nodes,
            error_node,
            cx,
        };
        this.check_rep();
        this
    }

    /// `step_size`: learning rate
    pub fn backpropagation_step<I>(&mut self, samples: &[I], step_size: f64)
    where
        I: AsRef<[f64]>,
    {
        let err = self.compute_error(samples, EvalOption::KeepCache);
        self.cx.buf().put(err);
        graph_do_gradient_descent_steps(&self.error_node, step_size, &mut self.cx);
        self.check_rep();
    }

    /// Return outputs from all terminal nodes
    ///
    /// `evaluate()[i]`: outputs from terminal node $i$
    pub fn evaluate<I>(&mut self, inputs: &[I]) -> Vec<Vec<f64>>
    where
        I: AsRef<[f64]>,
    {
        let mut outputs = vec![];

        for terminal_node in &self.terminal_nodes {
            let mut terminal_node = terminal_node.borrow_mut();
            terminal_node.evaluate_once(inputs, &mut self.cx);
            let output = terminal_node.output().unwrap();
            outputs.push(output.clone());
        }
        for terminal_node in &self.terminal_nodes {
            graph_delete_caches(terminal_node);
        }
        self.check_rep();
        outputs
    }

    pub fn error<S>(&mut self, dataset: &[S]) -> f64
    where
        S: AsRef<[f64]>,
    {
        let option = EvalOption::ClearCache;
        let mut progress_printer = ProgressPrinter::new();
        let mut error = 0.;
        for (i, inputs) in dataset.iter().enumerate() {
            let err = self.compute_error(&[inputs.as_ref()], option);
            error += err[0] / dataset.len() as f64;
            self.cx.buf().put(err);
            progress_printer.print_progress(i, dataset.len());
        }
        error
    }

    fn compute_error<I>(&mut self, inputs: &[I], option: EvalOption) -> Vec<f64>
    where
        I: AsRef<[f64]>,
    {
        let mut error_node = self.error_node.borrow_mut();
        error_node.evaluate_once(inputs, &mut self.cx);
        let mut output = self.cx.buf().take();
        output.extend(error_node.output().unwrap());
        drop(error_node);
        if matches!(option, EvalOption::ClearCache) {
            graph_delete_caches(&self.error_node);
        }
        self.check_rep();
        output
    }

    pub fn train<S>(&mut self, dataset: &[S], step_size: f64, max_steps: usize, option: TrainOption)
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
            self.backpropagation_step(&batch_input, step_size);
            progress_printer.print_progress(i, max_steps);
        }
        self.check_rep();
    }

    pub fn accuracy<S>(
        &mut self,
        dataset: &[S],
        accurate: impl Fn(AccurateFnParams<'_>) -> bool,
    ) -> f64
    where
        S: AsRef<[f64]>,
    {
        let mut progress_printer = ProgressPrinter::new();
        let mut accurate_count = 0;
        for (i, inputs) in dataset.iter().enumerate() {
            let eval = self.evaluate(&[inputs.as_ref()]);
            let eval = eval.iter().map(|x| x[0]).collect::<Vec<f64>>();
            let params = AccurateFnParams {
                inputs: inputs.as_ref(),
                outputs: eval,
            };
            if accurate(params) {
                accurate_count += 1;
            }
            progress_printer.print_progress(i, dataset.len());
        }
        self.check_rep();
        accurate_count as f64 / dataset.len() as f64
    }
}

#[derive(Debug, Clone)]
pub struct AccurateFnParams<'a> {
    pub inputs: &'a [f64],
    pub outputs: Vec<f64>,
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
