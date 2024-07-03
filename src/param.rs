use std::collections::HashMap;

use strict_num::NormalizedF64;
use vec_seg::{SegKey, VecSeg};

pub fn empty_shared_params() -> SegKey {
    SegKey::empty_slice()
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

    pub fn get_or_create_params<I>(self, create: impl Fn() -> I) -> SegKey
    where
        I: Iterator<Item = f64>,
    {
        self.injector.get_or_create_params(self.name, create)
    }
}

#[derive(Debug)]
pub struct ParamInjector {
    params: Params,
}
impl ParamInjector {
    pub fn new() -> Self {
        Self {
            params: Params::new(),
        }
    }

    pub fn get_or_create_params<I>(&mut self, name: String, create: impl Fn() -> I) -> SegKey
    where
        I: Iterator<Item = f64>,
    {
        if let Some(key) = self.params.key_by_name(&name) {
            return key;
        }
        self.params.try_insert_by_name(name, create)
    }

    pub fn into_params(self) -> Params {
        self.params
    }
}
impl Default for ParamInjector {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug)]
pub struct Params {
    seg: VecSeg<f64>,
    name: HashMap<String, SegKey>,
}
impl Params {
    pub fn new() -> Self {
        Self {
            seg: VecSeg::new(),
            name: HashMap::new(),
        }
    }

    pub fn len(&self) -> usize {
        self.name.len()
    }
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn try_insert_by_name<I>(&mut self, name: String, params: impl Fn() -> I) -> SegKey
    where
        I: Iterator<Item = f64>,
    {
        if let Some(&key) = self.name.get(&name) {
            return key;
        }
        let key = self.seg.extend(params());
        self.name.insert(name, key);
        key
    }

    pub fn seg(&self) -> &VecSeg<f64> {
        &self.seg
    }
    pub fn seg_mut(&mut self) -> &mut VecSeg<f64> {
        &mut self.seg
    }

    pub fn iter_name_key(&self) -> impl Iterator<Item = (&str, SegKey)> {
        self.name.iter().map(|(name, &key)| (name.as_str(), key))
    }
    pub fn key_by_name(&self, name: &str) -> Option<SegKey> {
        self.name.get(name).copied()
    }

    pub fn iter_name_slice(&self) -> impl Iterator<Item = (&str, &[f64])> {
        self.name
            .iter()
            .map(|(name, &key)| (name.as_str(), self.seg.slice(key)))
    }
    pub fn iter_name_slice_mut(&mut self, f: &mut impl FnMut(&str, &mut [f64])) {
        for (name, &key) in &self.name {
            let params = self.seg.slice_mut(key);
            f(name.as_str(), params);
        }
    }
    pub fn slice_by_name(&self, name: &str) -> Option<&[f64]> {
        let key = *self.name.get(name)?;
        Some(self.seg.slice(key))
    }
    pub fn slice_mut_by_name(&mut self, name: &str) -> Option<&mut [f64]> {
        let key = *self.name.get(name)?;
        Some(self.seg.slice_mut(key))
    }

    pub fn collect(&self) -> CollectedParams {
        let params = self
            .iter_name_slice()
            .map(|(name, slice)| (name.to_string(), slice.to_vec()));
        let params = HashMap::from_iter(params);
        CollectedParams { params }
    }
    pub fn overridden_by(&mut self, params: &CollectedParams) {
        self.iter_name_slice_mut(&mut |name, slice| {
            let params = params.params.get(name).unwrap();
            slice.copy_from_slice(params);
        });
    }
}
impl Default for Params {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone)]
pub struct CollectedParams {
    pub params: HashMap<String, Vec<f64>>,
}
impl genetic_algorithm::Dna for CollectedParams {
    fn mutate(&mut self, rate: NormalizedF64) {
        mutate(self, rate);
    }
}

pub fn crossover(a: &Params, b: &Params) -> CollectedParams {
    let mut collected = HashMap::new();
    assert_eq!(a.len(), b.len());
    let mut rng = rand::thread_rng();
    for (name, a) in a.iter_name_slice() {
        let b = b.slice_by_name(name).unwrap();
        assert_eq!(a.len(), b.len());
        let mut params = vec![];
        for (&a, &b) in a.iter().zip(b) {
            let param = genetic_algorithm::crossover(a, b, &mut rng);
            params.push(param);
        }
        collected.insert(name.to_owned(), params);
    }
    CollectedParams { params: collected }
}

pub fn mutate(params: &mut CollectedParams, rate: NormalizedF64) {
    let mut rng = rand::thread_rng();
    for params in params.params.values_mut() {
        for param in params {
            *param = genetic_algorithm::mutate(*param, rate, &mut rng);
        }
    }
}

#[cfg(test)]
pub mod tests {
    use std::{io::Write, path::Path};

    use super::*;

    pub fn read_params(path: impl AsRef<Path>) -> Option<CollectedParams> {
        let params_bin = std::fs::File::options().read(true).open(path).ok()?;
        let params: HashMap<String, Vec<f64>> = bincode::deserialize_from(params_bin).ok()?;
        Some(CollectedParams { params })
    }

    pub fn save_params(
        params: &CollectedParams,
        bin_path: impl AsRef<Path>,
        txt_path: impl AsRef<Path>,
    ) -> std::io::Result<()> {
        let txt = ron::to_string(&params.params).unwrap();
        let mut file = std::fs::File::options()
            .write(true)
            .create(true)
            .truncate(true)
            .open(txt_path)?;
        file.write_all(txt.as_bytes())?;

        let bin = bincode::serialize(&params.params).unwrap();
        let mut file = std::fs::File::options()
            .write(true)
            .create(true)
            .truncate(true)
            .open(bin_path)?;
        file.write_all(&bin)?;

        Ok(())
    }
}
