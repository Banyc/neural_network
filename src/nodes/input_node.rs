use super::node::{GeneralNode, NodeComputation};

pub fn input_node(input_index: usize) -> GeneralNode {
    let computation = InputNodeComputation { input_index };
    let node = GeneralNode::new(Vec::new(), Box::new(computation), Vec::new());
    node
}

struct InputNodeComputation {
    input_index: usize,
}

impl NodeComputation for InputNodeComputation {
    fn compute_output(
        &self,
        _parameters: &Vec<f64>,
        _operand_outputs: &Vec<f64>,
        inputs: &Vec<f64>,
    ) -> f64 {
        inputs[self.input_index]
    }

    fn compute_local_operand_gradient(
        &self,
        _parameters: &Vec<f64>,
        _operand_outputs: &Vec<f64>,
    ) -> Vec<f64> {
        Vec::new()
    }

    fn compute_local_parameter_gradient(
        &self,
        _parameters: &Vec<f64>,
        _operand_outputs: &Vec<f64>,
    ) -> Vec<f64> {
        Vec::new()
    }
}
