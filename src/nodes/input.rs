use crate::{
    computation::{NodeBackpropagationComputation, NodeComputation, NodeScalarComputation},
    mut_cell::MutCell,
    node::CompNode,
    param::empty_shared_params,
    ref_ctr::RefCtr,
};

/// ```math
/// f(x) = x[i]
/// ```
pub fn input_node(input_index: usize) -> CompNode {
    let computation = InputNodeComputation { input_index };
    CompNode::new(
        Vec::new(),
        RefCtr::new(MutCell::new(NodeComputation::Scalar(Box::new(computation)))),
        empty_shared_params(),
    )
}

#[derive(Debug, Clone)]
pub struct InputNodeGen {
    pub next_index: usize,
}
impl InputNodeGen {
    pub fn new() -> Self {
        Self { next_index: 0 }
    }

    pub fn gen(&mut self, len: usize) -> Vec<CompNode> {
        let start = self.next_index;
        self.next_index = start + len;
        input_node_batch(InputNodeBatchParams { start, len })
    }
}
impl Default for InputNodeGen {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone)]
pub struct InputNodeBatchParams {
    pub start: usize,
    pub len: usize,
}
pub fn input_node_batch(params: InputNodeBatchParams) -> Vec<CompNode> {
    (0..params.len)
        .map(|i| input_node(params.start + i))
        .collect()
}

#[derive(Debug)]
struct InputNodeComputation {
    input_index: usize,
}
impl NodeScalarComputation for InputNodeComputation {
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
}
impl NodeBackpropagationComputation for InputNodeComputation {
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

#[cfg(test)]
mod tests {
    use graph::dependency_order;

    use crate::{
        computation::{ComputationMode, NodeScalarComputation},
        node::{evaluate_once, GraphBuilder, NodeContext},
        param::Params,
    };

    use super::{input_node, InputNodeComputation};

    #[test]
    fn node_output1() {
        let mut graph = GraphBuilder::new();
        let node = graph.insert_node(input_node(0));
        let mut graph = graph.build();
        let nodes_forward = dependency_order(&graph, &[node]);
        let mut params = Params::new();
        let mut cx = NodeContext::new();
        evaluate_once(
            &mut graph,
            &nodes_forward,
            &mut params,
            &[&[3.0]],
            &mut cx,
            ComputationMode::Inference,
        );
        let node = graph.nodes().get(node).unwrap();
        let output = node.output().unwrap()[0];
        assert_eq!(output, 3.0);
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
