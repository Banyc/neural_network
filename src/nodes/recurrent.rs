use std::sync::{Arc, Mutex};

use graph::NodeIdx;

use crate::{
    computation::{NodeBackpropagationComputation, NodeComputation, NodeScalarComputation},
    node::CompNode,
    param::empty_shared_params,
};

pub fn recurrent_reader_node() -> (CompNode, Arc<Mutex<f64>>) {
    let value = Arc::new(Mutex::new(0.));
    let computation = RecurrentReaderNodeComputation {
        value: Arc::clone(&value),
    };
    let reader = CompNode::new(
        vec![],
        NodeComputation::Scalar(Box::new(computation)),
        empty_shared_params(),
    );
    (reader, value)
}
pub fn recurrent_writer_node(operand: NodeIdx, value: Arc<Mutex<f64>>) -> CompNode {
    let computation = RecurrentWriterNodeComputation { value };
    CompNode::new(
        vec![operand],
        NodeComputation::Scalar(Box::new(computation)),
        empty_shared_params(),
    )
}

#[derive(Debug)]
struct RecurrentReaderNodeComputation {
    value: Arc<Mutex<f64>>,
}
impl NodeScalarComputation for RecurrentReaderNodeComputation {
    fn compute_output(
        &self,
        parameters: &[f64],
        operand_outputs: &[f64],
        _graph_inputs: &[f64],
    ) -> f64 {
        assert!(parameters.is_empty());
        assert!(operand_outputs.is_empty());
        *self.value.lock().unwrap()
    }
}
impl NodeBackpropagationComputation for RecurrentReaderNodeComputation {
    fn compute_gradient_of_this_at_operand(
        &self,
        parameters: &[f64],
        operand_outputs: &[f64],
        buf: Vec<f64>,
    ) -> Vec<f64> {
        assert!(parameters.is_empty());
        assert!(operand_outputs.is_empty());
        buf
    }

    fn compute_gradient_of_this_at_parameter(
        &self,
        parameters: &[f64],
        operand_outputs: &[f64],
        buf: Vec<f64>,
    ) -> Vec<f64> {
        assert!(parameters.is_empty());
        assert!(operand_outputs.is_empty());
        buf
    }
}

#[derive(Debug)]
struct RecurrentWriterNodeComputation {
    value: Arc<Mutex<f64>>,
}
impl NodeScalarComputation for RecurrentWriterNodeComputation {
    fn compute_output(
        &self,
        parameters: &[f64],
        operand_outputs: &[f64],
        _graph_inputs: &[f64],
    ) -> f64 {
        assert!(parameters.is_empty());
        assert_eq!(operand_outputs.len(), 1);
        let x = operand_outputs[0];
        *self.value.lock().unwrap() = x;
        x
    }
}
impl NodeBackpropagationComputation for RecurrentWriterNodeComputation {
    fn compute_gradient_of_this_at_operand(
        &self,
        parameters: &[f64],
        operand_outputs: &[f64],
        mut buf: Vec<f64>,
    ) -> Vec<f64> {
        assert!(parameters.is_empty());
        assert_eq!(operand_outputs.len(), 1);
        buf.extend([1.]);
        buf
    }

    fn compute_gradient_of_this_at_parameter(
        &self,
        parameters: &[f64],
        operand_outputs: &[f64],
        buf: Vec<f64>,
    ) -> Vec<f64> {
        assert!(parameters.is_empty());
        assert_eq!(operand_outputs.len(), 1);
        buf
    }
}
