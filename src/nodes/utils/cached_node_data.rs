use std::sync::Arc;

/// The function of this node should be
/// $$
/// f : \mathbb{R}^n \to \mathbb{R}
/// $$
pub struct CachedNodeData {
    /// the output of this node
    pub output: Option<f64>,

    /// $$
    /// \frac{\partial E}{\partial f}
    /// $$
    ///
    /// - $E$: the out-most function of the entire network
    pub global_gradient: Option<f64>,

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
}

impl CachedNodeData {
    pub fn new() -> CachedNodeData {
        Self {
            output: None,
            global_gradient: None,
            local_operand_gradient: None,
            local_parameter_gradient: None,
            global_parameter_gradient: None,
        }
    }

    pub fn reset(&mut self) {
        self.output = None;
        self.global_gradient = None;
        self.local_operand_gradient = None;
        self.local_parameter_gradient = None;
        self.global_parameter_gradient = None;
    }
}
