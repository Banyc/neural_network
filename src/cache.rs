use crate::{node::NodeContext, two_d_slice::TwoDSlice, vec_segmenter::VecSegmenter};

#[derive(Debug, Clone)]
pub struct OperandOutputs<'a> {
    pub slice: &'a [f64],
    pub num_operands: usize,
}
impl<'a> OperandOutputs<'a> {
    pub fn get(self, batch_index: usize) -> &'a [f64] {
        let two_d = TwoDSlice::new(self.slice, self.num_operands);
        two_d.slice(batch_index)
    }
}

#[derive(Debug, Clone)]
pub struct NodeCacheBuilder {
    pub batch_size: usize,
    pub num_operands: usize,
    /// shape: (operands.len(), batch_size) : batch_size
    pub buf: Vec<f64>,
}
impl NodeCacheBuilder {
    pub fn build(mut self) -> NodeCache {
        let operand_outputs = self.batch_size * self.num_operands;
        let output = self.batch_size;
        let gradient_of_root_at_this = self.batch_size;
        assert_eq!(operand_outputs + output, self.buf.len());
        self.buf
            .extend(core::iter::repeat(0.).take(self.batch_size));
        let buf = VecSegmenter::new(
            self.buf,
            [operand_outputs, output, gradient_of_root_at_this],
        );
        NodeCache {
            buf,
            batch_size: self.batch_size,
            num_operands: self.num_operands,
            num_successors_distributed: 0,
        }
    }
}

#[derive(Debug, Clone)]
pub struct NodeCache {
    /// - shape: (operands.len(), batch_size) : batch_size : batch_size
    /// - contents:
    ///   1.  operand_outputs
    ///   1.  output
    ///   1.  sum_gradient_of_root_at_this
    buf: VecSegmenter<f64, 3>,
    batch_size: usize,
    num_operands: usize,
    num_successors_distributed: usize,
}
impl NodeCache {
    pub fn batch_size(&self) -> usize {
        self.batch_size
    }
    pub fn operand_outputs(&self, batch_index: usize) -> &[f64] {
        let slice = self.buf.segment(0).unwrap();
        OperandOutputs {
            slice,
            num_operands: self.num_operands,
        }
        .get(batch_index)
    }
    pub fn output(&self) -> &[f64] {
        self.buf.segment(1).unwrap()
    }
    pub fn backpropagate_mut(&mut self) -> BackpropagateCacheMut<'_> {
        let slice = self.buf.segment_mut(2).unwrap();
        BackpropagateCacheMut {
            sum_gradient_of_root_at_this: slice,
            num_successors_distributed: &mut self.num_successors_distributed,
        }
    }
    pub fn backpropagate(&self) -> BackpropagateCache<'_> {
        let slice = self.buf.segment(2).unwrap();
        BackpropagateCache {
            sum_gradient_of_root_at_this: slice,
            num_successors_distributed: self.num_successors_distributed,
        }
    }

    pub fn put_buf(self, cx: &mut NodeContext) {
        cx.buf().put(self.buf.into_vec())
    }
}

#[derive(Debug)]
pub struct BackpropagateCacheMut<'a> {
    sum_gradient_of_root_at_this: &'a mut [f64],
    num_successors_distributed: &'a mut usize,
}
impl BackpropagateCacheMut<'_> {
    pub fn add_up(&mut self, addend: f64, batch_index: usize) {
        self.sum_gradient_of_root_at_this[batch_index] += addend;
        if batch_index + 1 == self.sum_gradient_of_root_at_this.len() {
            *self.num_successors_distributed += 1;
        }
    }
}

#[derive(Debug)]
pub struct BackpropagateCache<'a> {
    sum_gradient_of_root_at_this: &'a [f64],
    num_successors_distributed: usize,
}
impl BackpropagateCache<'_> {
    /// ```math
    /// (\frac{\partial E}{\partial h_i} \cdot \frac{\partial h_i}{\partial f})
    /// ```
    ///
    /// - $h_i$: the $i$-th immediate successor of this node
    /// - $f$: this node
    pub fn gradient_of_root_at_this(&self, num_successors: usize) -> GradRootThis<'_> {
        assert!(self.num_successors_distributed <= num_successors);
        if num_successors == 0 {
            // this is the root node
            return GradRootThis::AllOnes;
        }
        if self.num_successors_distributed < num_successors {
            // FIXME: The output could be lead to terminal nodes that not covered by the error node
            return GradRootThis::NoEnoughAddends;
        }
        GradRootThis::Some(self.sum_gradient_of_root_at_this)
    }
}
pub enum GradRootThis<'a> {
    AllOnes,
    Some(&'a [f64]),
    NoEnoughAddends,
}
