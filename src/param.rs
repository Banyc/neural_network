use std::{collections::HashMap, sync::Arc};

use crate::mut_cell::MutCell;

pub type SharedParams = Arc<MutCell<Vec<f64>>>;
pub fn empty_shared_params() -> SharedParams {
    Arc::new(MutCell::new(vec![]))
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
            *params.borrow_mut() = p.clone();
        }
        self.params.insert(name, params);
    }

    pub fn collect_parameters(&self) -> HashMap<String, Vec<f64>> {
        let mut collected = HashMap::new();
        for (name, params) in &self.params {
            collected.insert(name.clone(), params.borrow().clone());
        }
        collected
    }
}

#[cfg(test)]
pub mod tests {
    use std::{io::Write, path::Path};

    use super::*;

    pub fn param_injector(path: impl AsRef<Path>) -> ParamInjector {
        let read_params = || {
            let params_bin = std::fs::File::options().read(true).open(path).ok()?;
            let params: HashMap<String, Vec<f64>> = bincode::deserialize_from(params_bin).ok()?;
            Some(params)
        };
        let params = read_params().unwrap_or_default();
        ParamInjector::new(params)
    }

    pub fn save_params(
        params: &HashMap<String, Vec<f64>>,
        bin_path: impl AsRef<Path>,
        txt_path: impl AsRef<Path>,
    ) -> std::io::Result<()> {
        let txt = ron::to_string(params).unwrap();
        let mut file = std::fs::File::options()
            .write(true)
            .create(true)
            .truncate(true)
            .open(txt_path)?;
        file.write_all(txt.as_bytes())?;

        let bin = bincode::serialize(params).unwrap();
        let mut file = std::fs::File::options()
            .write(true)
            .create(true)
            .truncate(true)
            .open(bin_path)?;
        file.write_all(&bin)?;

        Ok(())
    }
}
