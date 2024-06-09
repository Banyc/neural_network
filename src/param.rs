use std::{
    collections::HashMap,
    sync::{Arc, Mutex},
};

use crate::node::Node;

#[derive(Debug)]
pub struct ParamInjection<'a> {
    pub injector: &'a mut ParamInjector,
    pub name: String,
}
impl<'a> ParamInjection<'a> {
    pub fn name_append(&mut self, postfix: &str) -> ParamInjection {
        let mut name = self.name.clone();
        name.push_str(postfix);
        ParamInjection {
            injector: &mut *self.injector,
            name,
        }
    }

    pub fn insert_node(self, node: Arc<Mutex<Node>>) {
        self.injector.insert_node(self.name, node);
    }
}

#[derive(Debug)]
pub struct ParamInjector {
    params: HashMap<String, Vec<f64>>,
    nodes: HashMap<String, Arc<Mutex<Node>>>,
}
impl ParamInjector {
    pub fn new(parameters: HashMap<String, Vec<f64>>) -> Self {
        Self {
            params: parameters,
            nodes: HashMap::new(),
        }
    }
    pub fn empty() -> Self {
        Self {
            params: HashMap::new(),
            nodes: HashMap::new(),
        }
    }

    pub fn insert_node(&mut self, name: String, node: Arc<Mutex<Node>>) {
        assert!(!self.nodes.contains_key(&name));
        if let Some(p) = self.params.get(&name) {
            node.lock().unwrap().set_parameters(p.clone());
        }
        self.nodes.insert(name, node);
    }

    pub fn collect_parameters(&self) -> HashMap<String, Vec<f64>> {
        let mut params = HashMap::new();
        for (name, node) in &self.nodes {
            let node = node.lock().unwrap();
            let p = node.parameters();
            params.insert(name.clone(), p.clone());
        }
        params
    }
}
