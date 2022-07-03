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

#[cfg(test)]
mod tests {
    use super::{input_node, InputNodeComputation, NodeComputation};

    #[test]
    fn node_output1() {
        let mut node = input_node(0);
        let ret = node.evaluate(&vec![3.0]);
        assert!(ret >= 3.0);
        assert!(ret <= 3.0);
    }

    #[test]
    fn node_output2() {
        let mut node = input_node(0);
        let ret = node.evaluate(&vec![-4.0]);
        assert!(ret >= -4.0);
        assert!(ret <= -4.0);
    }

    #[test]
    fn computation_output() {
        let computation = InputNodeComputation { input_index: 0 };
        let ret = computation.compute_output(&vec![], &vec![], &vec![3.0]);
        assert!(ret >= 3.0);
        assert!(ret <= 3.0);
        let ret = computation.compute_output(&vec![], &vec![], &vec![-4.0]);
        assert!(ret >= -4.0);
        assert!(ret <= -4.0);
    }
}
