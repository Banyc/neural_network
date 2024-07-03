use graph::NodeIdx;

use crate::param::{crossover, CollectedParams};

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
impl genetic_algorithm::Agent for InferenceNetwork {
    type Dna = CollectedParams;
    fn crossover(&self, other: &Self) -> Self::Dna {
        let a = self.network().params();
        let b = other.network().params();
        crossover(a, b)
    }
    fn override_dna(&mut self, dna: Self::Dna) {
        self.network_mut().params_mut().overridden_by(&dna);
    }
}

pub type BrainPool = genetic_algorithm::Population<InferenceNetwork, CollectedParams>;

#[cfg(test)]
mod tests {
    use std::{
        num::NonZeroUsize,
        time::{Duration, Instant},
    };

    use rand::Rng;
    use strict_num::{FiniteF64, NormalizedF64};

    use crate::{
        layers::{activation::Activation, dense::dense_layer},
        node::GraphBuilder,
        nodes::{input::InputNodeGen, linear::LinearLayerConfig},
        param::{ParamInjection, ParamInjector},
    };

    use super::*;

    fn hidden_network(inputs: NonZeroUsize, outputs: NonZeroUsize) -> InferenceNetwork {
        let mut params_injector = ParamInjector::new();
        let mut params = ParamInjection {
            injector: &mut params_injector,
            name: "".into(),
        };
        let mut graph = GraphBuilder::new();
        let mut input_gen = InputNodeGen::new();
        let inputs = graph.insert_nodes(input_gen.gen(inputs.get()));
        let activation = Activation::ReLu;
        let dense = {
            let config = LinearLayerConfig {
                depth: NonZeroUsize::new(16).unwrap(),
                lambda: None,
            };
            let params = params.name_append(":hidden.0");
            let layer = dense_layer(&mut graph, inputs, config, &activation, params);
            graph.insert_nodes(layer)
        };
        let dense = {
            let config = LinearLayerConfig {
                depth: NonZeroUsize::new(8).unwrap(),
                lambda: None,
            };
            let params = params.name_append(":hidden.1");
            let layer = dense_layer(&mut graph, dense, config, &activation, params);
            graph.insert_nodes(layer)
        };
        let outputs = {
            let config = LinearLayerConfig {
                depth: outputs,
                lambda: None,
            };
            let params = params.name_append(":outputs");
            let layer = dense_layer(&mut graph, dense, config, &activation, params);
            graph.insert_nodes(layer)
        };
        let graph = graph.build();
        let params = params_injector.into_params();
        let nn = NeuralNetwork::new(graph, params);
        InferenceNetwork::new(nn, outputs)
    }

    #[ignore]
    #[test]
    fn train_genetic_xor() {
        let inputs = NonZeroUsize::new(2).unwrap();
        let outputs = NonZeroUsize::new(1).unwrap();
        let mutation_rate = NormalizedF64::new(0.01).unwrap();
        let mut individuals = vec![];
        let num_individuals = 128;
        for _ in 0..num_individuals {
            let nn = hidden_network(inputs, outputs);
            individuals.push(nn);
        }
        let mut brain_pool = BrainPool::new(individuals);
        let mut scores = vec![];
        let mut rng = rand::thread_rng();
        let mut last_print_time = Instant::now();
        loop {
            scores.clear();
            for brain in brain_pool.individuals_mut() {
                let mut score = 0.;
                let steps = 32;
                for _ in 0..steps {
                    let a: bool = rng.gen();
                    let b: bool = rng.gen();
                    let expected = a ^ b;
                    let buf = brain
                        .evaluate(&[&[a as u8 as f64, b as u8 as f64]])
                        .pop()
                        .unwrap();
                    let c = buf[0];
                    brain.network_mut().cx_mut().buf().put(buf);
                    let mse = (c - expected as u8 as f64).powi(2);
                    let s = if mse < 1. { 1. - mse } else { 0. };
                    score += s / steps as f64;
                }
                scores.push(score);
            }
            if Duration::from_secs(1) < last_print_time.elapsed() {
                let mut sorted = scores
                    .iter()
                    .map(|&x| FiniteF64::new(x).unwrap())
                    .collect::<Vec<FiniteF64>>();
                sorted.sort_unstable();
                println!();
                for score in sorted.iter().rev() {
                    print!("{:.2} ", score.get());
                }
                println!();
                last_print_time = Instant::now();
            }
            brain_pool.reproduce(&scores, mutation_rate);
        }
    }
}
