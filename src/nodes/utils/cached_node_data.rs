use std::sync::Arc;

/// The function of this node should be
/// $$
/// f : \mathbb{R}^n \to \mathbb{R}
/// $$
pub struct CachedNodeData {
    /// the output of this node
    pub output: Option<f64>,

    /// the outputs of the operands
    pub operand_outputs: Option<Arc<[f64]>>,

    /// $$
    /// \frac{\partial E}{\partial f}
    /// $$
    ///
    /// - $E$: the out-most function of the entire network
    pub gradient_of_root_at_function: Option<f64>,

    /// $$
    /// (\frac{\partial E}{\partial h_i} \cdot \frac{\partial h_i}{\partial f})
    /// $$
    ///
    /// - $h_i$: the $i$-th immediate successor of $f$
    pub addends_of_gradient_of_root_at_function: Vec<f64>,

    /// $$
    /// \frac{\partial f}{\partial z}
    /// $$
    ///
    /// - $z$: the non-tunable operands of $f$
    pub gradient_of_function_at_operand: Option<Arc<[f64]>>,

    /// $$
    /// \frac{\partial f}{\partial w}
    /// $$
    ///
    /// - $w$: the tunable parameters of $f$
    pub gradient_of_function_at_parameter: Option<Arc<[f64]>>,

    /// $$
    /// \frac{\partial E}{\partial w}
    /// $$
    ///
    /// - $w$: the tunable parameters of $f$
    pub gradient_of_root_at_parameter: Option<Arc<[f64]>>,

    /// Prevent from distributing it more than once
    pub has_distributed_addend_of_gradient_of_root_at_predecessor: bool,
}

impl CachedNodeData {
    pub fn new() -> CachedNodeData {
        Self {
            output: None,
            operand_outputs: None,
            gradient_of_root_at_function: None,
            addends_of_gradient_of_root_at_function: Vec::new(),
            gradient_of_function_at_operand: None,
            gradient_of_function_at_parameter: None,
            gradient_of_root_at_parameter: None,
            has_distributed_addend_of_gradient_of_root_at_predecessor: false,
        }
    }

    pub fn reset(&mut self) {
        self.output = None;
        self.operand_outputs = None;
        self.gradient_of_root_at_function = None;
        self.addends_of_gradient_of_root_at_function = Vec::new();
        self.gradient_of_function_at_operand = None;
        self.gradient_of_function_at_parameter = None;
        self.gradient_of_root_at_parameter = None;
        self.has_distributed_addend_of_gradient_of_root_at_predecessor = false;
    }
}

impl Default for CachedNodeData {
    fn default() -> Self {
        Self::new()
    }
}
