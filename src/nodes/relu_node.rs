use std::sync::{Arc, Mutex};

use super::node::{GeneralNode, NodeComputation};

pub fn relu_node(operand: Arc<Mutex<GeneralNode>>) -> GeneralNode {
    let computation = ReluNodeComputation {};
    let node = GeneralNode::new(vec![operand], Box::new(computation), Vec::new());
    node
}

struct ReluNodeComputation {}

impl NodeComputation for ReluNodeComputation {
    fn compute_output(
        &self,
        _parameters: &Vec<f64>,
        operand_outputs: &Vec<f64>,
        _inputs: &Vec<f64>,
    ) -> f64 {
        assert_eq!(operand_outputs.len(), 1);
        relu(operand_outputs[0])
    }

    fn compute_local_operand_gradient(
        &self,
        _parameters: &Vec<f64>,
        operand_outputs: &Vec<f64>,
    ) -> Vec<f64> {
        assert_eq!(operand_outputs.len(), 1);
        vec![relu_derivative(operand_outputs[0])]
    }

    fn compute_local_parameter_gradient(
        &self,
        _parameters: &Vec<f64>,
        _operand_outputs: &Vec<f64>,
    ) -> Vec<f64> {
        Vec::new()
    }
}

fn relu(x: f64) -> f64 {
    f64::max(x, 0.0)
}

fn relu_derivative(x: f64) -> f64 {
    match x {
        _ if x > 0.0 => 1.0,
        _ => 0.0,
    }
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Mutex};

    use super::super::{input_node::input_node, relu_node::relu_node};

    #[test]
    fn evaluate_negative() {
        let input_node = input_node(0);
        let mut relu = relu_node(Arc::new(Mutex::new(input_node)));
        let ret = relu.evaluate(&vec![-2.0]);
        assert!(ret >= 0.0);
        assert!(ret <= 0.0);
    }

    #[test]
    fn evaluate_positive() {
        let input_node = input_node(0);
        let mut relu = relu_node(Arc::new(Mutex::new(input_node)));
        let ret = relu.evaluate(&vec![3.0]);
        assert!(ret >= 3.0);
        assert!(ret <= 3.0);
    }

    #[test]
    fn local_operand_gradient_positive() {
        let input_node = input_node(0);
        let mut relu = relu_node(Arc::new(Mutex::new(input_node)));
        relu.evaluate(&vec![3.0]);
        let ret = relu.local_operand_gradient().unwrap();
        assert!(ret[0] >= 1.0);
        assert!(ret[0] <= 1.0);
    }

    #[test]
    fn local_operand_gradient_negative() {
        let input_node = input_node(0);
        let mut relu = relu_node(Arc::new(Mutex::new(input_node)));
        relu.evaluate(&vec![-3.0]);
        let ret = relu.local_operand_gradient().unwrap();
        assert!(ret[0] >= 0.0);
        assert!(ret[0] <= 0.0);
    }

    #[test]
    fn local_parameter_gradient_empty() {
        let input_node = input_node(0);
        let mut relu = relu_node(Arc::new(Mutex::new(input_node)));
        relu.evaluate(&vec![3.0]);
        let ret = relu.local_parameter_gradient().unwrap();
        assert_eq!(ret.len(), 0);
    }
}
