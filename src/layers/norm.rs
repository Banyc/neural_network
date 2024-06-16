use crate::{
    node::SharedNode,
    nodes::{
        batch_norm::{batch_norm_layer, BatchNormLayerConfig},
        layer_norm::layer_norm_layer,
    },
    param::ParamInjection,
};

#[derive(Clone)]
pub enum Normalization {
    BatchNorm { config: BatchNormLayerConfig },
    LayerNorm,
}
impl Normalization {
    pub fn normalize(
        self,
        inputs: Vec<SharedNode>,
        param_injection: ParamInjection<'_>,
    ) -> Vec<SharedNode> {
        match self {
            Normalization::BatchNorm { config } => {
                batch_norm_layer(inputs, config, param_injection)
            }
            Normalization::LayerNorm => layer_norm_layer(inputs),
        }
    }
}
