use std::collections::HashSet;

use rand::Rng;

pub struct SamplingConfig {
    pub temperature: f32,
    pub top_k: usize,
    pub top_p: f32,
}

impl SamplingConfig {
    /// Exactly the branch used by `sample`; NaN temperature remains stochastic.
    pub fn is_greedy(&self) -> bool {
        self.temperature <= 0.0
    }

    pub const DEFAULT: Self = Self {
        temperature: 0.0,
        top_k: 64,
        top_p: 0.95,
    };
}

pub fn sample(logits: &[f32], config: &SamplingConfig, banned: &HashSet<usize>, rng: &mut impl Rng) -> usize {
    assert!(!logits.is_empty(), "sample requires at least one logit");
    if config.is_greedy() {
        return argmax(logits, banned);
    }
    let temperature = config.temperature;

    let mut ranked: Vec<usize> = (0..logits.len()).collect();
    if config.top_k > 0 {
        let keep = config.top_k.saturating_add(banned.len()).min(ranked.len());
        ranked.select_nth_unstable_by(keep - 1, |&a, &b| logits[b].total_cmp(&logits[a]));
        ranked.truncate(keep);
    }
    ranked.sort_unstable_by(|&a, &b| logits[b].total_cmp(&logits[a]));

    if !banned.is_empty() {
        let filtered: Vec<usize> = ranked.iter().copied().filter(|id| !banned.contains(id)).collect();
        if !filtered.is_empty() {
            ranked = filtered;
        }
    }
    if config.top_k > 0 {
        ranked.truncate(config.top_k.min(ranked.len()));
    }

    let max_logit = logits[ranked[0]];
    let mut probs: Vec<f32> = ranked
        .iter()
        .map(|&i| ((logits[i] - max_logit) / temperature).exp())
        .collect();
    let sum: f32 = probs.iter().sum();
    for p in &mut probs {
        *p /= sum;
    }

    let mut nucleus_len = probs.len();
    let mut cumulative = 0.0;
    for (i, &p) in probs.iter().enumerate() {
        cumulative += p;
        if cumulative >= config.top_p {
            nucleus_len = i + 1;
            break;
        }
    }
    let nucleus_sum: f32 = probs[..nucleus_len].iter().sum();

    let threshold = rng.random::<f32>() * nucleus_sum;
    let mut acc = 0.0;
    for (i, &p) in probs[..nucleus_len].iter().enumerate() {
        acc += p;
        if acc >= threshold {
            return ranked[i];
        }
    }
    ranked[nucleus_len - 1]
}

/// Highest logit outside `banned`; falls back to the overall highest if every token is banned.
fn argmax(logits: &[f32], banned: &HashSet<usize>) -> usize {
    let mut best: Option<usize> = None;
    for (index, value) in logits.iter().enumerate() {
        // A value that cannot beat the current allowed maximum cannot affect
        // selection, so it needs no hash lookup. Strict comparison keeps first-ID ties.
        if best.is_none_or(|previous| value.total_cmp(&logits[previous]).is_gt())
            && !banned.contains(&index)
        {
            best = Some(index);
        }
    }
    best.unwrap_or_else(|| {
        (0..logits.len()).max_by(|&a, &b| logits[a].total_cmp(&logits[b]).then(b.cmp(&a))).unwrap()
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    fn reference(logits: &[f32], banned: &HashSet<usize>) -> usize {
        let maximum = |allowed: &dyn Fn(usize)->bool| (0..logits.len()).filter(|&i|allowed(i))
            .max_by(|&a,&b|logits[a].total_cmp(&logits[b]).then(b.cmp(&a)));
        maximum(&|i|!banned.contains(&i)).or_else(||maximum(&|_|true)).unwrap()
    }
    #[test]
    fn greedy_matches_reference_for_all_bf16_patterns_and_arbitrary_bans() {
        let mut values: Vec<_> = (0..=u16::MAX).map(crate::host_logit_bytes::bf16_bits_to_f32).collect();
        for order in 0..3 {
            if order == 1 {values.reverse();}
            if order == 2 {values.sort_unstable_by_key(|x|x.to_bits().wrapping_mul(2654435761));}
            for bans in [HashSet::new(), (0..values.len()).step_by(3).collect(), (0..values.len()).collect(), [0,1,65535,999999].into_iter().collect()] {
                assert_eq!(argmax(&values,&bans),reference(&values,&bans));
            }
        }
    }
    #[test]
    fn greedy_keeps_signed_zero_nan_order_and_lowest_token_ties() {
        for values in [vec![-0.0,0.0,0.0],vec![1.0,1.0,1.0],vec![f32::NEG_INFINITY,-2.0,f32::INFINITY],vec![f32::from_bits(0xffc00003),f32::from_bits(0x7fc00001),f32::from_bits(0x7fc00003)]] {
            for mask in 0..8 {
                let banned=(0..values.len()).filter(|i|mask&(1<<i)!=0).collect();
                assert_eq!(argmax(&values,&banned),reference(&values,&banned));
            }
        }
        assert_eq!(argmax(&[-0.0,0.0,0.0],&HashSet::new()),1);
        assert_eq!(argmax(&[1.0,1.0,1.0],&HashSet::new()),0);
    }
}
