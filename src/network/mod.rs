use std::time::{Duration, Instant};

use graph::{dependency_order, Graph, NodeIdx};
use rand::Rng;

use crate::{
    computation::ComputationMode,
    node::{backpropagate, delete_cache, evaluate_once, CompNode, NodeContext},
    param::Params,
};

pub mod inference;
pub mod train;

const _: fn() = || {
    fn assert_send<T: Sync + Send + 'static>() {}
    assert_send::<NeuralNetwork>();
};

#[derive(Debug)]
pub struct NeuralNetwork {
    graph: Graph<CompNode>,
    params: Params,
    /// global necessity
    cx: NodeContext,
}
impl NeuralNetwork {
    fn check_rep(&self) {}

    pub fn new(graph: Graph<CompNode>, params: Params) -> NeuralNetwork {
        let cx = NodeContext::new();
        let this = NeuralNetwork { graph, params, cx };
        this.check_rep();
        this
    }

    pub fn graph(&self) -> &Graph<CompNode> {
        &self.graph
    }
    pub fn params(&self) -> &Params {
        &self.params
    }
    pub fn params_mut(&mut self) -> &mut Params {
        &mut self.params
    }
    pub fn cx_mut(&mut self) -> &mut NodeContext {
        &mut self.cx
    }

    /// `step_size`: learning rate
    pub fn compute_error_and_backpropagate<I>(
        &mut self,
        error: &GraphOrder,
        samples: &[I],
        step_size: f64,
    ) where
        I: AsRef<[f64]>,
    {
        let err = self.compute_error(
            error,
            samples,
            EvalOption::KeepCache,
            ComputationMode::Training,
        );
        self.cx.buf().put(err);
        backpropagate(
            &mut self.graph,
            error.forward(),
            &mut self.params,
            step_size,
            &mut self.cx,
        );
        self.check_rep();
    }

    /// Return outputs from all terminal nodes
    ///
    /// `evaluate()[i]`: outputs from terminal node $i$
    pub fn evaluate<I>(&mut self, terminals: &GraphOrder, inputs: &[I]) -> Vec<Vec<f64>>
    where
        I: AsRef<[f64]>,
    {
        let mut outputs = vec![];
        evaluate_once(
            &mut self.graph,
            terminals.forward(),
            &mut self.params,
            inputs,
            &mut self.cx,
            ComputationMode::Inference,
        );
        for &terminal_node in terminals.destinations() {
            let terminal_node = self.graph.nodes().get(terminal_node).unwrap();
            let output = terminal_node.output().unwrap();
            outputs.push(output.to_vec());
        }
        delete_cache(&mut self.graph, terminals.forward(), &mut self.cx);
        self.check_rep();
        outputs
    }

    pub fn compute_avg_error<S>(&mut self, error: &GraphOrder, dataset: &[S]) -> f64
    where
        S: AsRef<[f64]>,
    {
        let option = EvalOption::ClearCache;
        let mut progress_printer = ProgressPrinter::new();
        let mut avg = 0.;
        for (i, inputs) in dataset.iter().enumerate() {
            let err = self.compute_error(
                error,
                &[inputs.as_ref()],
                option,
                ComputationMode::Inference,
            );
            avg += err[0] / dataset.len() as f64;
            self.cx.buf().put(err);
            progress_printer.print_progress(i, dataset.len());
        }
        avg
    }

    fn compute_error<I>(
        &mut self,
        error: &GraphOrder,
        inputs: &[I],
        option: EvalOption,
        mode: ComputationMode,
    ) -> Vec<f64>
    where
        I: AsRef<[f64]>,
    {
        evaluate_once(
            &mut self.graph,
            error.forward(),
            &mut self.params,
            inputs,
            &mut self.cx,
            mode,
        );
        let mut output = self.cx.buf().take();
        let error_node = self.graph.nodes().get(error.only_destination()).unwrap();
        output.extend(error_node.output().unwrap());
        match option {
            EvalOption::KeepCache => (),
            EvalOption::ClearCache => {
                delete_cache(&mut self.graph, error.forward(), &mut self.cx);
            }
        }
        self.check_rep();
        output
    }

    pub fn train<S>(
        &mut self,
        error: &GraphOrder,
        dataset: &[S],
        step_size: f64,
        max_steps: usize,
        option: TrainOption,
    ) where
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
            self.compute_error_and_backpropagate(error, &batch_input, step_size);
            progress_printer.print_progress(i, max_steps);
        }
        self.check_rep();
    }

    pub fn accuracy<S>(
        &mut self,
        terminals: &GraphOrder,
        dataset: &[S],
        accurate: impl Fn(AccurateFnParams<'_>) -> bool,
    ) -> f64
    where
        S: AsRef<[f64]>,
    {
        let mut progress_printer = ProgressPrinter::new();
        let mut accurate_count = 0;
        for (i, inputs) in dataset.iter().enumerate() {
            let eval = self.evaluate(terminals, &[inputs.as_ref()]);
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

#[derive(Debug, Clone)]
pub struct GraphOrder {
    destinations: Vec<NodeIdx>,
    forward: Vec<NodeIdx>,
}
impl GraphOrder {
    pub fn new<T>(graph: &Graph<T>, destinations: Vec<NodeIdx>) -> Self
    where
        T: graph::Node,
    {
        let forward = dependency_order(graph, &destinations);
        Self {
            destinations,
            forward,
        }
    }

    pub fn only_destination(&self) -> NodeIdx {
        assert_eq!(self.destinations.len(), 1);
        self.destinations[0]
    }
    pub fn destinations(&self) -> &[NodeIdx] {
        &self.destinations
    }
    pub fn forward(&self) -> &[NodeIdx] {
        &self.forward
    }
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
