use std::{
    collections::HashMap,
    sync::{Arc, Mutex},
};

pub type SharedParams = Arc<Mutex<Vec<f64>>>;
pub fn empty_shared_params() -> SharedParams {
    Arc::new(Mutex::new(vec![]))
}

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

    pub fn insert_params(self, params: SharedParams) {
        self.injector.insert_params(self.name, params);
    }
}

#[derive(Debug)]
pub struct ParamInjector {
    prev_params: HashMap<String, Vec<f64>>,
    params: HashMap<String, SharedParams>,
}
impl ParamInjector {
    pub fn new(parameters: HashMap<String, Vec<f64>>) -> Self {
        Self {
            prev_params: parameters,
            params: HashMap::new(),
        }
    }
    pub fn empty() -> Self {
        Self {
            prev_params: HashMap::new(),
            params: HashMap::new(),
        }
    }

    pub fn insert_params(&mut self, name: String, params: SharedParams) {
        assert!(!self.params.contains_key(&name));
        if let Some(p) = self.prev_params.get(&name) {
            *params.lock().unwrap() = p.clone();
        }
        self.params.insert(name, params);
    }

    pub fn collect_parameters(&self) -> HashMap<String, Vec<f64>> {
        let mut collected = HashMap::new();
        for (name, params) in &self.params {
            collected.insert(name.clone(), params.lock().unwrap().clone());
        }
        collected
    }
}
