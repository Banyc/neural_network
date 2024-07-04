use strict_num::FiniteF64;

use crate::network::AccurateFnParams;

mod mnist;
mod nasdaq;
mod neural_network;

pub fn multi_class_one_hot_accurate(params: AccurateFnParams<'_>, classes: usize) -> bool {
    let eval = params.outputs;
    let label = &params.inputs[params.inputs.len() - classes..];
    assert_eq!(eval.len(), label.len());
    assert!(!eval.is_empty());
    let eval_max_i = max_i(&eval);
    let label_max_i = max_i(label);
    eval_max_i == label_max_i
}

pub fn max_i(x: &[f64]) -> usize {
    assert!(!x.is_empty());
    let mut max = x[0];
    let mut max_i = 0;
    for (i, x) in x.iter().copied().enumerate() {
        FiniteF64::new(x).unwrap();
        if max < x {
            max = x;
            max_i = i;
        }
    }
    max_i
}
