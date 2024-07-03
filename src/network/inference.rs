use graph::NodeIdx;

use super::{AccurateFnParams, GraphOrder, NeuralNetwork};

#[derive(Debug)]
pub struct InferenceNetwork {
    network: NeuralNetwork,
    /// output: a prediction
    terminals: GraphOrder,
}
impl InferenceNetwork {
    fn check_rep(&self) {}

    pub fn new(network: NeuralNetwork, terminal_nodes: Vec<NodeIdx>) -> Self {
        let terminals = GraphOrder::new(network.graph(), terminal_nodes);
        let this = Self { network, terminals };
        this.check_rep();
        this
    }

    pub fn network(&self) -> &NeuralNetwork {
        &self.network
    }
    pub fn network_mut(&mut self) -> &mut NeuralNetwork {
        &mut self.network
    }

    /// Return outputs from all terminal nodes
    ///
    /// `evaluate()[i]`: outputs from terminal node $i$
    pub fn evaluate<I>(&mut self, inputs: &[I]) -> Vec<Vec<f64>>
    where
        I: AsRef<[f64]>,
    {
        self.network.evaluate(&self.terminals, inputs)
    }

    pub fn accuracy<S>(
        &mut self,
        dataset: &[S],
        accurate: impl Fn(AccurateFnParams<'_>) -> bool,
    ) -> f64
    where
        S: AsRef<[f64]>,
    {
        self.network.accuracy(&self.terminals, dataset, accurate)
    }
}
