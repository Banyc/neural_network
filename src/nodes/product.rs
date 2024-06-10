use std::sync::Arc;

use crate::{
    node::{Node, NodeComputation, SharedNode},
    param::empty_shared_params,
};

/// ```math
/// f(x) = \prod x
/// ```
pub fn product_node(operands: Vec<SharedNode>) -> Node {
    let computation = ProductNodeComputation {};
    Node::new(operands, Arc::new(computation), empty_shared_params())
}

#[derive(Debug)]
struct ProductNodeComputation {}
impl NodeComputation for ProductNodeComputation {
    fn compute_output(
        &self,
        parameters: &[f64],
        operand_outputs: &[f64],
        _graph_inputs: &[f64],
    ) -> f64 {
        assert!(parameters.is_empty());
        product(operand_outputs)
    }

    fn compute_gradient_of_this_at_operand(
        &self,
        parameters: &[f64],
        operand_outputs: &[f64],
    ) -> Vec<f64> {
        assert!(parameters.is_empty());
        product_derivative(operand_outputs)
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

fn product(x: &[f64]) -> f64 {
    x.iter().product()
}

fn product_derivative(x: &[f64]) -> Vec<f64> {
    let mut der = vec![];
    for i in 0..x.len() {
        let d = product_except(x, i);
        der.push(d);
    }
    der
}
fn product_except(x: &[f64], ind: usize) -> f64 {
    let mut prod = 1.;
    for (i, x) in x.iter().enumerate() {
        if i == ind {
            continue;
        }
        prod *= x;
    }
    prod
}