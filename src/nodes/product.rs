use crate::{
    computation::{NodeBackpropagationComputation, NodeComputation, NodeScalarComputation},
    mut_cell::MutCell,
    node::{Node, SharedNode},
    param::empty_shared_params,
    ref_ctr::RefCtr,
};

/// ```math
/// f(x) = \prod x
/// ```
pub fn product_node(operands: Vec<SharedNode>) -> Node {
    let computation = ProductNodeComputation {};
    Node::new(
        operands,
        RefCtr::new(MutCell::new(NodeComputation::Scalar(Box::new(computation)))),
        empty_shared_params(),
    )
}

#[derive(Debug)]
struct ProductNodeComputation {}
impl NodeScalarComputation for ProductNodeComputation {
    fn compute_output(
        &self,
        parameters: &[f64],
        operand_outputs: &[f64],
        _graph_inputs: &[f64],
    ) -> f64 {
        assert!(parameters.is_empty());
        assert!(!operand_outputs.is_empty());
        product(operand_outputs)
    }
}
impl NodeBackpropagationComputation for ProductNodeComputation {
    fn compute_gradient_of_this_at_operand(
        &self,
        parameters: &[f64],
        operand_outputs: &[f64],
        buf: Vec<f64>,
    ) -> Vec<f64> {
        assert!(parameters.is_empty());
        assert!(!operand_outputs.is_empty());
        product_derivative(operand_outputs, buf)
    }

    fn compute_gradient_of_this_at_parameter(
        &self,
        parameters: &[f64],
        operand_outputs: &[f64],
        buf: Vec<f64>,
    ) -> Vec<f64> {
        assert!(parameters.is_empty());
        assert!(!operand_outputs.is_empty());
        buf
    }
}

fn product(x: &[f64]) -> f64 {
    x.iter().product()
}

fn product_derivative(x: &[f64], mut buf: Vec<f64>) -> Vec<f64> {
    for i in 0..x.len() {
        let d = product_except(x, i);
        buf.push(d);
    }
    buf
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
