use std::collections::HashMap;

use rand::Rng;
use rand_distr::Distribution;
use strict_num::NormalizedF64;

use crate::{mut_cell::MutCell, ref_ctr::RefCtr};

pub type SharedParams = RefCtr<MutCell<Vec<f64>>>;
pub fn empty_shared_params() -> SharedParams {
    RefCtr::new(MutCell::new(vec![]))
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

    pub fn get_or_create_params(self, create: impl Fn() -> SharedParams) -> SharedParams {
        self.injector.get_or_create_params(self.name, create)
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

    pub fn get_or_create_params(
        &mut self,
        name: String,
        create: impl Fn() -> SharedParams,
    ) -> SharedParams {
        if let Some(params) = self.params.get(&name) {
            return RefCtr::clone(params);
        }
        let params = create();
        if let Some(p) = self.prev_params.get(&name) {
            *params.borrow_mut() = p.clone();
        }
        self.params.insert(name, RefCtr::clone(&params));
        params
    }

    pub fn collect_parameters(&self) -> HashMap<String, Vec<f64>> {
        let mut collected = HashMap::new();
        for (name, params) in &self.params {
            collected.insert(name.clone(), params.borrow().clone());
        }
        collected
    }
}

pub fn crossover(a: &HashMap<String, SharedParams>, b: &HashMap<String, SharedParams>) {
    assert_eq!(a.len(), b.len());
    let mut coin_flip = rand::thread_rng();
    for (k, a) in a.iter() {
        let b = b.get(k).unwrap().borrow();
        assert_eq!(a.borrow().len(), b.len());
        for (a, b) in a.borrow_mut().iter_mut().zip(b.iter().copied()) {
            if coin_flip.gen_bool(0.5) {
                continue;
            }
            *a = b;
        }
    }
}

pub fn mutate(params: &HashMap<String, SharedParams>, rate: NormalizedF64) {
    let mut rng = rand::thread_rng();
    let normal = rand_distr::Normal::new(0., 1.).unwrap();
    for (_, params) in params.iter() {
        for param in params.borrow_mut().iter_mut() {
            let chance = rng.gen_bool(rate.get());
            if !chance {
                continue;
            }
            let change = normal.sample(&mut rng);
            *param += change;
            *param = param.clamp(-1., 1.);
        }
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
