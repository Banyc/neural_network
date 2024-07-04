use std::{
    io::BufReader,
    num::NonZeroUsize,
    path::Path,
    time::{Duration, Instant},
};

use anyhow::Context;
use rand::Rng;
use serde::Deserialize;
use strict_num::{FiniteF64, NormalizedF64};

use crate::{
    network::inference::{tests::hidden_network, BrainPool, InferenceNetwork},
    param::tests::{read_params, save_params},
};

const HISTORICAL_QUOTES_DIR: &str = "local/nasdaq/historical_quotes";
const PARAMS_BIN: &str = "local/nasdaq/params.bincode";
const PARAMS_TXT: &str = "local/nasdaq/params.ron";

#[ignore]
#[test]
fn train_genetic_stock() {
    let params = read_params(PARAMS_BIN);

    let stocks = read_quotes(HISTORICAL_QUOTES_DIR).unwrap();
    let dataset = Dataset::new(stocks);

    let inputs = NonZeroUsize::new(6).unwrap();
    let outputs = NonZeroUsize::new(1).unwrap();
    let mut prev = vec![0.; inputs.get()];
    let mut future = vec![0.; 1];

    let mutation_rate = NormalizedF64::new(0.01).unwrap();
    let mut individuals = vec![];
    let num_individuals = 128;
    for _ in 0..num_individuals {
        let mut nn = hidden_network(
            inputs,
            outputs,
            &[32, 16]
                .into_iter()
                .map(|x| NonZeroUsize::new(x).unwrap())
                .collect::<Vec<NonZeroUsize>>(),
        );
        if let Some(params) = &params {
            nn.network_mut().params_mut().overridden_by(params);
        }
        individuals.push(nn);
    }
    let mut brain_pool = BrainPool::new(individuals);
    let mut scores = vec![];
    let mut rng = rand::thread_rng();
    let mut last_print_time = Instant::now();

    #[derive(Debug)]
    struct Episode {
        pub profit: f64,
        pub score: f64,
    }
    let mut run_episode = |brain: &mut InferenceNetwork| {
        let mut episode = Episode {
            profit: 0.,
            score: 0.,
        };
        let steps = 128;
        for _ in 0..steps {
            dataset.select(&mut prev, &mut future, &mut rng);
            let buf = brain.evaluate(&[&prev]).pop().unwrap();
            let action = buf[0];
            brain.network_mut().cx_mut().buf().put(buf);

            let bet = action.clamp(0., 1.);
            let change = future[0];
            let profit = bet * change;
            let score = profit.powi(2) * profit.signum();

            // println!("action: {action}; bet: {bet}; change: {change}; score: {score}");

            episode.score += score / steps as f64;
            episode.profit += profit / steps as f64;
        }
        episode
    };

    let mut portfolio = 0.;
    loop {
        scores.clear();
        for brain in brain_pool.individuals_mut() {
            let res = run_episode(brain);
            scores.push(FiniteF64::new(res.score).unwrap());
        }
        if Duration::from_secs(1) < last_print_time.elapsed() {
            let sorted = {
                let mut sorted = scores.clone();
                sorted.sort_unstable();
                sorted
            };
            let best_score = *sorted.last().unwrap();
            let best_brain = {
                let i = scores.iter().position(|&x| x == best_score).unwrap();
                &mut brain_pool.individuals_mut()[i]
            };
            save_params(
                &best_brain.network().params().collect(),
                PARAMS_BIN,
                PARAMS_TXT,
            )
            .unwrap();
            // {
            //     println!();
            //     for score in sorted.iter().rev() {
            //         print!("{:.2} ", score.get());
            //     }
            //     println!();
            // }
            {
                let res = run_episode(best_brain);
                portfolio += res.profit;
                println!("[best] score: {best_score}; test: {res:?}; portfolio: {portfolio}");
            }
            last_print_time = Instant::now();
        }
        brain_pool.reproduce(&scores, mutation_rate);
    }
}

#[derive(Debug)]
pub struct Dataset {
    stocks: Vec<Stock>,
}
impl Dataset {
    pub fn new(stocks: Vec<Stock>) -> Self {
        Self { stocks }
    }

    pub fn select(&self, prev: &mut [f64], future: &mut [f64], mut rng: impl Rng) {
        let i = rng.gen_range(0..self.stocks.len());
        let stock = &self.stocks[i];

        let size = prev.len() + future.len() + 1;
        let prev_start = rng.gen_range(0..stock.quotes.len().checked_sub(size).unwrap());
        let pivot = prev_start + prev.len();
        let future_start = pivot + 1;

        let pivot = stock.quotes[pivot].close;

        for (i, x) in stock.quotes[prev_start..]
            .iter()
            .take(prev.len())
            .enumerate()
        {
            prev[i] = change_rate(x.close as i64, pivot as i64);
        }
        for (i, x) in stock.quotes[future_start..]
            .iter()
            .take(future.len())
            .enumerate()
        {
            future[i] = change_rate(x.close as i64, pivot as i64);
        }
    }
}
fn change_rate(value: i64, anchor: i64) -> f64 {
    let change = value - anchor;
    change as f64 / anchor as f64
}

#[derive(Debug)]
pub struct Stock {
    pub quotes: Vec<Quote>,
}
#[derive(Debug)]
pub struct Quote {
    pub close: usize,
}
/// src: <https://www.nasdaq.com/market-activity/quotes/historical>
#[derive(Debug, Deserialize)]
struct HistoricalQuoteRaw {
    #[serde(rename = "Close/Last")]
    pub close: String,
}
pub fn read_quotes(dir: impl AsRef<Path>) -> anyhow::Result<Vec<Stock>> {
    let dir = std::fs::read_dir(dir)?;
    let mut stocks = vec![];
    for entry in dir {
        let entry = entry?;
        let path = entry.path();
        let Some(ext) = path.extension() else {
            continue;
        };
        if ext != "csv" {
            continue;
        }
        let file = std::fs::File::options().read(true).open(path)?;
        let read_buf = BufReader::new(file);
        let mut csv_rdr = csv::Reader::from_reader(read_buf);
        let mut quotes = vec![];
        for raw_quote in csv_rdr.deserialize() {
            let raw_quote: HistoricalQuoteRaw = raw_quote?;
            let close = parse_dollar(&raw_quote.close, 2).context("close")?;
            let quote = Quote { close };
            quotes.push(quote);
        }
        let stock = Stock { quotes };
        stocks.push(stock);
    }
    Ok(stocks)
}
fn parse_dollar(dollar: &str, precision: usize) -> Option<usize> {
    let dollar = dollar.trim();
    if !dollar.starts_with("$") {
        return None;
    }
    let decimal = &dollar[1..];
    let value = parse_decimal(decimal, precision)?;
    Some(value)
}
fn parse_decimal(decimal: &str, precision: usize) -> Option<usize> {
    let (integer, fraction) = match decimal.trim().split_once(".") {
        Some(x) => x,
        None => (decimal, ""),
    };
    let integer: usize = integer.parse().ok()?;
    let integer = integer * usize::pow(10, u32::try_from(precision).ok()?);
    let fraction = if fraction.is_empty() {
        0
    } else {
        let fractions_taken = fraction.len().min(precision);
        let padding = precision - fractions_taken;
        let fraction = &fraction[..fractions_taken];
        let fraction: usize = fraction.parse().ok()?;
        fraction * usize::pow(10, u32::try_from(padding).ok()?)
    };
    Some(integer + fraction)
}
#[test]
fn test_parse_decimal() {
    assert_eq!(parse_decimal("123.45", 1).unwrap(), 1234);
    assert_eq!(parse_decimal("123.45", 3).unwrap(), 123450);
    assert_eq!(parse_decimal("123", 1).unwrap(), 1230);
}
