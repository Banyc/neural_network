use graph::NodeIdx;

use super::{inference::InferenceNetwork, GraphOrder, TrainOption};

#[derive(Debug)]
pub struct TrainNetwork {
    inference: InferenceNetwork,
    /// output: the error between the prediction and the label
    error: GraphOrder,
}
impl TrainNetwork {
    fn check_rep(&self) {}

    pub fn new(inference: InferenceNetwork, error_node: NodeIdx) -> Self {
        let error = GraphOrder::new(inference.network().graph(), vec![error_node]);
        let this = Self { inference, error };
        this.check_rep();
        this
    }

    pub fn inference(&self) -> &InferenceNetwork {
        &self.inference
    }
    pub fn inference_mut(&mut self) -> &mut InferenceNetwork {
        &mut self.inference
    }

    /// `step_size`: learning rate
    pub fn compute_error_and_backpropagate<I>(&mut self, samples: &[I], step_size: f64)
    where
        I: AsRef<[f64]>,
    {
        self.inference
            .network_mut()
            .compute_error_and_backpropagate(&self.error, samples, step_size)
    }

    pub fn compute_avg_error<S>(&mut self, dataset: &[S]) -> f64
    where
        S: AsRef<[f64]>,
    {
        self.inference
            .network_mut()
            .compute_avg_error(&self.error, dataset)
    }

    pub fn train<S>(&mut self, dataset: &[S], step_size: f64, max_steps: usize, option: TrainOption)
    where
        S: AsRef<[f64]>,
    {
        self.inference
            .network_mut()
            .train(&self.error, dataset, step_size, max_steps, option)
    }
}
