use graph::NodeIdx;

use crate::{
    node::{CompNode, GraphBuilder},
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
        graph: &mut GraphBuilder,
        inputs: Vec<NodeIdx>,
        param_injection: ParamInjection<'_>,
    ) -> Vec<CompNode> {
        match self {
            Normalization::BatchNorm { config } => {
                batch_norm_layer(inputs, config, param_injection)
            }
            Normalization::LayerNorm => layer_norm_layer(graph, inputs),
        }
    }
}
