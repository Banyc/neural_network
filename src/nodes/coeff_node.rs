use std::sync::{Arc, Mutex};

use super::node::{Node, NodeComputation};

/// ```math
/// f(x) = ax
/// ```
pub fn coeff_node(operand: Arc<Mutex<Node>>, coefficient: f64) -> Node {
    let computation = Arc::new(CoeffNodeComputation { coefficient });
    Node::new(vec![operand], computation, vec![])
}

#[derive(Debug)]
struct CoeffNodeComputation {
    coefficient: f64,
}
impl NodeComputation for CoeffNodeComputation {
    fn compute_output(
        &self,
        parameters: &[f64],
        operand_outputs: &[f64],
        _graph_inputs: &[f64],
    ) -> f64 {
        assert!(parameters.is_empty());
        assert_eq!(operand_outputs.len(), 1);
        self.coefficient * operand_outputs[0]
    }

    fn compute_gradient_of_this_at_operand(
        &self,
        parameters: &[f64],
        operand_outputs: &[f64],
    ) -> Vec<f64> {
        assert!(parameters.is_empty());
        assert_eq!(operand_outputs.len(), 1);
        vec![self.coefficient]
    }

    fn compute_gradient_of_this_at_parameter(
        &self,
        parameters: &[f64],
        operand_outputs: &[f64],
    ) -> Vec<f64> {
        assert!(parameters.is_empty());
        assert_eq!(operand_outputs.len(), 1);
        vec![]
    }
}
