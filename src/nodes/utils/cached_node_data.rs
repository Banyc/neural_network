use std::sync::Arc;

/// The function of this node should be
/// $$
/// f : \mathbb{R}^n \to \mathbb{R}
/// $$
pub struct CachedNodeData {
    /// the output of this node
    pub output: Option<f64>,

    /// the outputs of the operands
    pub operand_outputs: Option<Arc<Vec<f64>>>,

    /// $$
    /// \frac{\partial E}{\partial f}
    /// $$
    ///
    /// - $E$: the out-most function of the entire network
    pub global_gradient: Option<f64>,

    /// $$
    /// (\frac{\partial E}{\partial h_i} \cdot \frac{\partial h_i}{\partial f})
    /// $$
    ///
    /// - $h_i$: the $i$-th immediate successor of $f$
    pub global_gradient_entries: Vec<f64>,

    /// $$
    /// \frac{\partial f}{\partial z}
    /// $$
    ///
    /// - $z$: the non-tunable operands of $f$
    pub local_operand_gradient: Option<Arc<Vec<f64>>>,

    /// $$
    /// \frac{\partial f}{\partial w}
    /// $$
    ///
    /// - $w$: the tunable parameters of $f$
    pub local_parameter_gradient: Option<Arc<Vec<f64>>>,

    /// $$
    /// \frac{\partial E}{\partial w}
    /// $$
    ///
    /// - $w$: the tunable parameters of $f$
    pub global_parameter_gradient: Option<Arc<Vec<f64>>>,

    /// Prevent from distributing it more than once
    pub has_distributed_global_gradient_entries: bool,
}

impl CachedNodeData {
    pub fn new() -> CachedNodeData {
        Self {
            output: None,
            operand_outputs: None,
            global_gradient: None,
            global_gradient_entries: Vec::new(),
            local_operand_gradient: None,
            local_parameter_gradient: None,
            global_parameter_gradient: None,
            has_distributed_global_gradient_entries: false,
        }
    }

    pub fn reset(&mut self) {
        self.output = None;
        self.operand_outputs = None;
        self.global_gradient = None;
        self.global_gradient_entries = Vec::new();
        self.local_operand_gradient = None;
        self.local_parameter_gradient = None;
        self.global_parameter_gradient = None;
        self.has_distributed_global_gradient_entries = false;
    }
}

impl Default for CachedNodeData {
    fn default() -> Self {
        Self::new()
    }
}
