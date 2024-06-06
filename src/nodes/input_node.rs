use std::sync::{Arc, Mutex};

use crate::node::{Node, NodeComputation};

/// ```math
/// f(x) = x[i]
/// ```
pub fn input_node(input_index: usize) -> Node {
    let computation = InputNodeComputation { input_index };
    Node::new(Vec::new(), Arc::new(computation), Vec::new())
}

pub fn input_node_batch(len: usize) -> Vec<Arc<Mutex<Node>>> {
    (0..len)
        .map(|i| {
            let node = input_node(i);
            Arc::new(Mutex::new(node))
        })
        .collect()
}

#[derive(Debug)]
struct InputNodeComputation {
    input_index: usize,
}
impl NodeComputation for InputNodeComputation {
    fn compute_output(
        &self,
        parameters: &[f64],
        operand_outputs: &[f64],
        graph_inputs: &[f64],
    ) -> f64 {
        assert!(parameters.is_empty());
        assert!(operand_outputs.is_empty());
        graph_inputs[self.input_index]
    }

    fn compute_gradient_of_this_at_operand(
        &self,
        parameters: &[f64],
        operand_outputs: &[f64],
    ) -> Vec<f64> {
        assert!(parameters.is_empty());
        assert!(operand_outputs.is_empty());
        Vec::new()
    }

    fn compute_gradient_of_this_at_parameter(
        &self,
        parameters: &[f64],
        operand_outputs: &[f64],
    ) -> Vec<f64> {
        assert!(parameters.is_empty());
        assert!(operand_outputs.is_empty());
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
