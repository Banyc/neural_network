use std::sync::{Arc, Mutex};

use rand::Rng;

use super::node::{GeneralNode, NodeComputation};

pub fn weight_node(operands: Vec<Arc<Mutex<GeneralNode>>>) -> GeneralNode {
    let computation = WeightNodeComputation {};
    let mut weights = Vec::new();
    let op_len = operands.len();
    let weight_bound = 1.0 / (op_len as f64).sqrt();
    let mut rng = rand::thread_rng();
    for _ in 0..op_len {
        let weight: f64 = rng.gen_range(-weight_bound..weight_bound);
        weights.push(weight);
    }
    let node = GeneralNode::new(operands, Box::new(computation), weights);
    node
}

struct WeightNodeComputation {}

impl NodeComputation for WeightNodeComputation {
    fn compute_output(
        &mut self,
        parameters: &Vec<f64>,
        operand_outputs: &Vec<f64>,
        _inputs: &Vec<f64>,
    ) -> f64 {
        assert_eq!(operand_outputs.len(), 1);
        assert_eq!(parameters.len(), 1);
        weight(operand_outputs, parameters)
    }

    fn compute_local_operand_gradient(
        &self,
        parameters: &Vec<f64>,
        _operand_outputs: &Vec<f64>,
    ) -> Vec<f64> {
        weight_derivative(parameters)
    }

    fn compute_local_parameter_gradient(
        &self,
        _parameters: &Vec<f64>,
        operand_outputs: &Vec<f64>,
    ) -> Vec<f64> {
        derivative_of_w(operand_outputs)
    }

    fn reset(&mut self) {}
}

fn weight(x: &Vec<f64>, w: &Vec<f64>) -> f64 {
    assert_eq!(x.len(), w.len());
    let mut sum = 0.0;
    for i in 0..w.len() {
        sum += x[i] * w[i];
    }
    sum
}

fn weight_derivative(w: &Vec<f64>) -> Vec<f64> {
    w.clone()
}

fn derivative_of_w(x: &Vec<f64>) -> Vec<f64> {
    x.clone()
}
