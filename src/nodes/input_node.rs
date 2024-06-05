use std::{cell::RefCell, rc::Rc};

use super::node::{Node, NodeComputation};

pub fn input_node(input_index: usize) -> Node {
    let computation = InputNodeComputation { input_index };
    Node::new(Vec::new(), Rc::new(computation), Vec::new())
}

pub fn input_node_batch(len: usize) -> Vec<Rc<RefCell<Node>>> {
    (0..len)
        .map(|i| {
            let node = input_node(i);
            Rc::new(RefCell::new(node))
        })
        .collect()
}

#[derive(Debug)]
struct InputNodeComputation {
    input_index: usize,
}
impl NodeComputation for InputNodeComputation {
    fn compute_output(&self, _parameters: &[f64], _operand_outputs: &[f64], inputs: &[f64]) -> f64 {
        inputs[self.input_index]
    }

    fn compute_gradient_of_this_at_operand(
        &self,
        _parameters: &[f64],
        _operand_outputs: &[f64],
    ) -> Vec<f64> {
        Vec::new()
    }

    fn compute_gradient_of_this_at_parameter(
        &self,
        _parameters: &[f64],
        _operand_outputs: &[f64],
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
        let ret = node.evaluate_once(&[3.0]);
        assert_eq!(ret, 3.0);
    }

    #[test]
    fn node_output2() {
        let mut node = input_node(0);
        let ret = node.evaluate_once(&[-4.0]);
        assert_eq!(ret, -4.0);
    }

    #[test]
    fn computation_output() {
        let computation = InputNodeComputation { input_index: 0 };
        let ret = computation.compute_output(&[], &[], &[3.0]);
        assert_eq!(ret, 3.0);
        let ret = computation.compute_output(&[], &[], &[-4.0]);
        assert_eq!(ret, -4.0);
    }
}
