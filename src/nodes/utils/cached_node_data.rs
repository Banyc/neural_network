use std::rc::Rc;

/// The function of this node should be
/// ```math
/// f : \mathbb{R}^n \to \mathbb{R}
/// ```
#[derive(Debug, Clone)]
pub struct CachedNodeData {
    /// the output of this node
    pub output: Option<f64>,

    /// the outputs of the operands
    pub operand_outputs: Option<Rc<[f64]>>,

    /// ```math
    /// \frac{\partial E}{\partial f}
    /// ```
    ///
    /// - $E$: the out-most function of the entire network
    pub partial_derivative_of_root_at_this: Option<f64>,

    /// ```math
    /// (\frac{\partial E}{\partial h_i} \cdot \frac{\partial h_i}{\partial f})
    /// ```
    ///
    /// - $h_i$: the $i$-th immediate successor of $f$
    pub addends_of_gradient_of_root_at_this: Vec<f64>,

    /// ```math
    /// \frac{\partial f}{\partial z}
    /// ```
    ///
    /// - $z$: the non-tunable operands of $f$
    pub gradient_of_this_at_operand: Option<Rc<[f64]>>,

    /// ```math
    /// \frac{\partial f}{\partial w}
    /// ```
    ///
    /// - $w$: the tunable parameters of $f$
    pub gradient_of_this_at_parameter: Option<Rc<[f64]>>,

    /// ```math
    /// \frac{\partial E}{\partial w}
    /// ```
    ///
    /// - $w$: the tunable parameters of $f$
    pub gradient_of_root_at_parameter: Option<Rc<[f64]>>,

    /// Prevent from distributing it more than once
    pub has_distributed_addends_of_partial_derivatives_of_root_at_operands_to_operands: bool,
}
impl CachedNodeData {
    pub fn new() -> CachedNodeData {
        Self {
            output: None,
            operand_outputs: None,
            partial_derivative_of_root_at_this: None,
            addends_of_gradient_of_root_at_this: Vec::new(),
            gradient_of_this_at_operand: None,
            gradient_of_this_at_parameter: None,
            gradient_of_root_at_parameter: None,
            has_distributed_addends_of_partial_derivatives_of_root_at_operands_to_operands: false,
        }
    }

    pub fn reset(&mut self) {
        self.output = None;
        self.operand_outputs = None;
        self.partial_derivative_of_root_at_this = None;
        self.addends_of_gradient_of_root_at_this.clear();
        self.gradient_of_this_at_operand = None;
        self.gradient_of_this_at_parameter = None;
        self.gradient_of_root_at_parameter = None;
        self.has_distributed_addends_of_partial_derivatives_of_root_at_operands_to_operands = false;
    }
}
impl Default for CachedNodeData {
    fn default() -> Self {
        Self::new()
    }
}
