use super::node::{GeneralNode, NodeComputation};

pub fn bias_node(operand: GeneralNode) -> GeneralNode {
    let computation = BiasNodeComputation {};
    let node = GeneralNode::new(vec![operand], Box::new(computation), vec![0.0]);
    node
}

struct BiasNodeComputation {}

impl NodeComputation for BiasNodeComputation {
    fn compute_output(
        &self,
        parameters: &Vec<f64>,
        operand_outputs: &Vec<f64>,
        _inputs: &Vec<f64>,
    ) -> f64 {
        assert_eq!(operand_outputs.len(), 1);
        assert_eq!(parameters.len(), 1);
        bias(operand_outputs[0], parameters[0])
    }

    fn compute_local_operand_gradient(
        &self,
        _parameters: &Vec<f64>,
        operand_outputs: &Vec<f64>,
    ) -> Vec<f64> {
        assert_eq!(operand_outputs.len(), 1);
        vec![bias_derivative()]
    }

    fn compute_local_parameter_gradient(
        &self,
        _parameters: &Vec<f64>,
        _operand_outputs: &Vec<f64>,
    ) -> Vec<f64> {
        vec![1.0]
    }
}

fn bias(x: f64, b: f64) -> f64 {
    x + b
}

fn bias_derivative() -> f64 {
    1.0
}
