//! Model hyperparameters read from GGUF metadata, plus the per-architecture table.

use crate::gguf::Gguf;
use anyhow::{Context, bail};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LayerKind { Delta, Attn }

/// Standalone EAGLE3 1-layer draft head parameters.
#[derive(Clone, Debug)]
pub struct Eagle3Config {
    pub extract_layers: [usize; 3],
    pub n_ff: usize,
    pub n_head: usize,
    pub n_head_kv: usize,
    pub head_dim: usize,
    pub n_rot: usize,
    pub rope_base: f32,
    pub eps: f32,
    pub draft_vocab: usize,
    pub norm_before_residual: bool,
    pub rope_neox: bool,
    pub d2t: Vec<i32>,
}

/// What differs between the supported architectures; everything else comes from GGUF metadata.
struct Arch {
    /// DeltaNet layers with full attention every `full_attention_interval` layers; otherwise all attention.
    hybrid: bool,
    /// `attn_q` is [query | sigmoid output gate] per head (2 * head_dim wide).
    q_gate: bool,
    /// DeltaNet value head h reads key head h / (nv / nk) (llama.cpp qwen3next), not h % nk.
    k_interleave: bool,
    /// Canonical tensor suffix to this architecture's suffix.
    aliases: &'static [(&'static str, &'static str)],
}

fn arch(name: &str) -> Option<Arch> {
    const FFN_NORM: &[(&str, &str)] = &[("post_attention_norm.weight", "ffn_norm.weight")];
    Some(match name {
        "qwen35" | "qwen35moe" => Arch { hybrid: true, q_gate: true, k_interleave: false, aliases: &[] },
        // Qwen3.8-Flash-Next: qwen35-style DeltaNet pairing, plus hyper-connections and PLE read from metadata.
        "qwen4exp" => Arch { hybrid: true, q_gate: true, k_interleave: false, aliases: &[] },
        "qwen3next" => Arch { hybrid: true, q_gate: true, k_interleave: true, aliases: &[] },
        // Recursive Flint: Qwen-style attention with sliding windows, per-head QK norms and attention sinks.
        "flint" => Arch { hybrid: false, q_gate: false, k_interleave: false, aliases: FFN_NORM },
        "qwen3" | "qwen3moe" => Arch { hybrid: false, q_gate: false, k_interleave: false, aliases: FFN_NORM },
        _ => return None,
    })
}

/// YaRN rope scaling from the GGUF metadata (`rope.scaling.type = yarn`).
#[derive(Clone, Copy, Debug)]
pub struct Yarn {
    /// Context extension: the low-frequency dimensions are compressed by this factor.
    pub factor: f32,
    /// The context the model was trained on before the extension (`rope.scaling.original_context_length`).
    pub orig_ctx: usize,
    /// The frequency boundaries of the interpolation ramp (`yarn_beta_fast`, `yarn_beta_slow`).
    pub beta_fast: f32,
    pub beta_slow: f32,
}
/// How many trailing layers are MTP draft layers. `nextn_predict_layers` can outlive the draft head tensors (a
/// finetune that drops the head keeps the base model's metadata), so the declared count is kept only when the
/// tensors of the first draft layer are there; the `Err` carries the warning to print. `block_count` includes the
/// draft layers, so the real layer count subtracts them either way.
fn nextn_present(declared: usize, block_count: usize, has: impl Fn(&str) -> bool) -> (usize, usize, Result<(), String>) {
    if declared == 0 || declared >= block_count { return (block_count, 0, Ok(())); }
    let first = block_count - declared;
    if has(&format!("blk.{first}.nextn.eh_proj.weight")) { return (first, declared, Ok(())); }
    (first, 0, Err(format!("no MTP draft tensors (blk.{first}.nextn.*): loading without the draft head, though the \
        metadata declares nextn_predict_layers={declared}")))
}

/// The per-pair rope frequencies (`theta = pos * freq[i]`, `n_rot / 2` of them) for `base`, with YaRN applied when
/// `yarn` is set: the long-wavelength pairs are interpolated towards a context `factor` times longer, with a ramp
/// between `beta_fast` and `beta_slow` so the short wavelengths keep their resolution. Same arithmetic as
/// llama.cpp's `rope_yarn` and `rope_yarn_corr_dims` (the attention scale factor is 1 there and here).
pub fn rope_freqs_of(base: f32, n_rot: usize, yarn: Option<Yarn>) -> Vec<f32> {
    let n = n_rot.max(2);
    let plain = |i: usize| base.powf(-2.0 * i as f32 / n as f32);
    let Some(y) = yarn.filter(|y| y.factor > 0.0 && y.orig_ctx > 0) else {
        return (0..n / 2).map(plain).collect();
    };
    // The pair indices where the interpolation ramp starts and ends. Small `i` (short wavelengths) keep the
    // original frequency, large `i` (long wavelengths) are compressed by `factor`, and the ramp blends them.
    let cd = |rot: f32| (n as f32 * (y.orig_ctx as f32 / (rot * 2.0 * std::f32::consts::PI)).ln() / (2.0 * base.ln()))
        .floor().max(0.0);
    let lo = cd(y.beta_fast);
    let hi = cd(y.beta_slow).ceil().min(n as f32 - 1.0);
    (0..n / 2).map(|i| {
        let extrap = plain(i);
        let interp = extrap / y.factor;
        let mix = 1.0 - ((i as f32 - lo) / (hi - lo).max(0.001)).clamp(0.0, 1.0);
        interp + (extrap - interp) * mix
    }).collect()
}

#[derive(Clone, Debug)]
pub struct ModelConfig {
    pub arch: String,
    pub n_layer: usize,
    pub n_nextn: usize,
    pub n_embd: usize,
    pub n_head: usize,
    pub n_head_kv: usize,
    pub head_dim: usize,
    pub n_rot: usize,
    /// Qwen3-VL/3.5 M-RoPE section sizes (sum = `n_rot/2`); `None` uses 1-D RoPE.
    pub mrope_sections: Option<[usize; 4]>,
    pub rope_base: f32,
    /// YaRN rope scaling (`{arch}.rope.scaling.type = yarn`): the frequencies of the interpolated dimensions are
    /// divided by `factor`, with the usual ramp between `beta_fast` and `beta_slow`.
    pub rope_yarn: Option<Yarn>,
    /// `rope_freq()` for every pair (`n_rot / 2` values), computed once at load: the kernels and the CPU path
    /// both rotate with `theta = pos * rope_freqs[i]`.
    pub rope_freqs: Vec<f32>,
    pub eps: f32,
    pub n_vocab: usize,
    /// Dense FFN size (0 for MoE).
    pub n_ff: usize,
    pub n_expert: usize,
    pub n_expert_used: usize,
    pub n_ff_exp: usize,
    pub n_ff_shexp: usize,
    /// Leading layers with a dense FFN; the rest are MoE (0 when the whole model is MoE).
    pub n_dense: usize,
    /// 0 = no extra scale.
    pub expert_weights_scale: f32,
    /// Router activation is sigmoid (Flint) instead of softmax.
    pub router_sigmoid: bool,
    /// Sliding-window width for the layers flagged in `swa` (0 = all layers are dense).
    pub n_swa: usize,
    /// Per-layer sliding-window attention (Flint). Empty for architectures without it.
    pub swa: Vec<bool>,
    /// Per-layer RoPE; Flint's full-attention layers use no positional encoding.
    pub rope_layer: Vec<bool>,
    /// Per-head learned attention sinks (`blk.N.attn_sinks.weight`).
    pub attn_sinks: bool,
    /// DeltaNet sizes, 0 without DeltaNet layers.
    pub ssm_d_inner: usize,
    pub ssm_d_state: usize,
    pub ssm_n_group: usize,
    pub ssm_dt_rank: usize,
    pub ssm_conv: usize,
    /// Hyper-connection residual streams and their low rank (qwen4exp; 0 = plain residual).
    pub hc_n: usize,
    pub hc_rank: usize,
    /// Per-layer (n-gram) embedding width (qwen4exp; 0 = none).
    pub n_embd_ple: usize,
    /// QSA sparse attention indexer (qwen4exp): heads, head size, and the number of cells kept per row
    /// (0 = dense attention).
    pub idx_n_head: usize,
    pub idx_head_dim: usize,
    pub idx_top_k: usize,
    /// Mixer of each main layer (MTP layers are attention).
    pub layers: Vec<LayerKind>,
    pub q_gate: bool,
    pub k_interleave: bool,
    pub aliases: &'static [(&'static str, &'static str)],
    pub eagle3: Option<Eagle3Config>,
}

impl ModelConfig {
    pub fn from_gguf(g: &Gguf) -> anyhow::Result<ModelConfig> {
        let arch = g.get("general.architecture")?.as_str().context("general.architecture")?.to_string();
        let Some(a) = self::arch(&arch) else {
            bail!("unsupported architecture '{arch}' (supported: qwen3, qwen3moe, qwen3next, qwen35, qwen35moe, qwen4exp, flint)")
        };
        let opt = |k: &str| g.meta.get(&format!("{arch}.{k}")).and_then(|v| v.as_u64()).unwrap_or(0) as usize;
        let req = |k: &str| -> anyhow::Result<usize> {
            let key = format!("{arch}.{k}");
            Ok(g.get(&key)?.as_u64().with_context(|| format!("{key}: not an integer"))? as usize)
        };
        let f = |k: &str| g.meta.get(&format!("{arch}.{k}")).and_then(|v| v.as_f32());
        // A finetuned model can keep `nextn_predict_layers` from its base while dropping the draft head's tensors:
        // trusting the metadata would then load layer `block_count - n_nextn` from tensors that are not there.
        let block_count = req("block_count")?;
        let (n_layer, n_nextn, missing) = nextn_present(opt("nextn_predict_layers"), block_count,
            |n| g.tensors.contains_key(n));
        if let Err(n) = missing { eprintln!("{n}"); }
        let layers: Vec<LayerKind> = if a.hybrid {
            let every = match opt("full_attention_interval") { 0 => 4, v => v };
            (0..n_layer).map(|il| if (il + 1) % every == 0 { LayerKind::Attn } else { LayerKind::Delta }).collect()
        } else {
            vec![LayerKind::Attn; n_layer]
        };
        let has_delta = layers.contains(&LayerKind::Delta);
        let ssm = |k: &str| if has_delta { req(k) } else { Ok(opt(k)) };
        let (n_embd, n_head) = (req("embedding_length")?, req("attention.head_count")?);
        let head_dim = match opt("attention.key_length") { 0 => n_embd / n_head, v => v };
        let n_expert = opt("expert_count");
        anyhow::ensure!(n_expert <= 512, "{n_expert} experts (at most 512 supported)");
        let n_dense = opt("leading_dense_block_count");
        anyhow::ensure!(n_dense <= n_layer, "{n_dense} leading dense blocks for {n_layer} layers");
        let flags = |k: &str| -> anyhow::Result<Option<Vec<bool>>> {
            let key = format!("{arch}.{k}");
            let Some(v) = g.meta.get(&key) else { return Ok(None) };
            let a = v.as_arr().with_context(|| format!("{key}: not an array"))?;
            anyhow::ensure!(a.len() >= n_layer, "{key} has {} entries for {n_layer} layers", a.len());
            Ok(Some(a[..n_layer].iter().map(|v| v.as_u64().unwrap_or(0) != 0).collect()))
        };
        let swa = flags("attention.sliding_window_pattern")?.unwrap_or_else(|| vec![false; n_layer]);
        let rope_layer = flags("attention.rope_pattern")?.unwrap_or_else(|| vec![true; n_layer]);
        let n_swa = opt("attention.sliding_window");
        anyhow::ensure!(!swa.contains(&true) || n_swa > 0, "sliding-window layers without attention.sliding_window");
        let emb = g.tensors.get("token_embd.weight").context("missing tensor token_embd.weight")?;
        let mut c = ModelConfig {
            n_layer,
            n_nextn,
            n_embd,
            n_head,
            n_head_kv: req("attention.head_count_kv")?,
            head_dim,
            n_rot: match opt("rope.dimension_count") { 0 => head_dim, v => v },
            mrope_sections: g.meta.get(&format!("{arch}.rope.dimension_sections")).and_then(|v| v.as_arr()).and_then(|a| {
                if a.len() < 4 { return None; }
                let mut s = [0usize; 4];
                for (o, v) in s.iter_mut().zip(a) { *o = v.as_u64()? as usize; }
                Some(s)
            }),
            rope_base: f("rope.freq_base").unwrap_or(10000.0),
            rope_yarn: match g.meta.get(&format!("{arch}.rope.scaling.type")).and_then(|v| v.as_str()) {
                Some("yarn") => Some(Yarn {
                    factor: f("rope.scaling.factor").unwrap_or(1.0),
                    orig_ctx: opt("rope.scaling.original_context_length"),
                    beta_fast: f("rope.scaling.yarn_beta_fast").unwrap_or(32.0),
                    beta_slow: f("rope.scaling.yarn_beta_slow").unwrap_or(1.0),
                }),
                Some(t) => bail!("rope scaling '{t}' is not supported (only yarn)"),
                None => None,
            },
            eps: f("attention.layer_norm_rms_epsilon").context("missing rms epsilon")?,
            n_vocab: emb.dims[1] as usize,
            n_ff: opt("feed_forward_length"),
            n_expert,
            n_expert_used: opt("expert_used_count"),
            n_ff_exp: opt("expert_feed_forward_length"),
            n_ff_shexp: opt("expert_shared_feed_forward_length"),
            n_dense,
            expert_weights_scale: f("expert_weights_scale").unwrap_or(0.0),
            router_sigmoid: arch == "flint" && opt("expert_gating_func") == 2,
            n_swa,
            swa,
            rope_layer,
            attn_sinks: g.tensors.contains_key("blk.0.attn_sinks.weight"),
            ssm_d_inner: ssm("ssm.inner_size")?,
            ssm_d_state: ssm("ssm.state_size")?,
            ssm_n_group: ssm("ssm.group_count")?,
            ssm_dt_rank: ssm("ssm.time_step_rank")?,
            ssm_conv: ssm("ssm.conv_kernel")?,
            hc_n: opt("hyper_connection.count"),
            hc_rank: opt("hyper_connection.low_rank"),
            n_embd_ple: opt("embedding_length_per_layer_input"),
            idx_n_head: opt("attention.indexer.head_count"),
            idx_head_dim: opt("attention.indexer.key_length"),
            idx_top_k: opt("attention.indexer.top_k"),
            layers,
            q_gate: a.q_gate,
            k_interleave: a.k_interleave,
            aliases: a.aliases,
            arch,
            eagle3: None,
            rope_freqs: vec![],
        };
        // Kernels divide by these and index value heads by nv / nk key heads.
        let (nk, nv) = (c.ssm_n_group, c.ssm_dt_rank);
        anyhow::ensure!(!has_delta || (nk > 0 && nv > 0 && nv % nk == 0 && c.ssm_d_inner % nv == 0),
            "bad ssm dims: group_count {nk}, time_step_rank {nv}, inner_size {}", c.ssm_d_inner);
        c.rope_freqs = c.rope_freq();
        Ok(c)
    }

    /// The per-pair rope frequencies: `theta = pos * rope_freq()[i]` for pair `i` of `n_rot / 2`. Plain RoPE is
    /// `base^(-2i/n_rot)`; with YaRN the low-frequency (long-wavelength) pairs are interpolated towards the
    /// frequencies of a context `factor` times longer, with a ramp between `beta_fast` and `beta_slow` so the short
    /// wavelengths keep their resolution. Same arithmetic as llama.cpp's `rope_yarn`.
    pub fn rope_freq(&self) -> Vec<f32> { rope_freqs_of(self.rope_base, self.n_rot, self.rope_yarn) }

    pub fn is_moe(&self) -> bool { self.n_expert > 0 }
    /// Layer `il` uses the MoE FFN (dense-lead architectures start with `n_dense` dense layers).
    pub fn is_moe_layer(&self, il: usize) -> bool { self.n_expert > 0 && il >= self.n_dense }
    /// DeltaNet value head size (0 without DeltaNet layers).
    pub fn ssm_dv(&self) -> usize { self.ssm_d_inner / self.ssm_dt_rank.max(1) }
}

#[cfg(test)]
mod rope_freq_tests {
    use super::*;

    /// The draft head's metadata is trusted only when its tensors are there, and the real layer count never
    /// includes the draft layers either way.
    #[test]
    fn a_missing_draft_head_is_not_loaded() {
        let has = |n: &str| n == "blk.40.nextn.eh_proj.weight";
        assert_eq!(nextn_present(1, 41, has), (40, 1, Ok(())));
        assert!(nextn_present(1, 41, |_| false).2.is_err(), "a head without tensors must say so");
        assert_eq!(nextn_present(1, 41, |_| false).0, 40, "the draft layer is not a real layer");
        assert_eq!(nextn_present(1, 41, |_| false).1, 0, "no draft head to load");
        assert_eq!(nextn_present(0, 41, has), (41, 0, Ok(())));
        assert_eq!(nextn_present(41, 41, has), (41, 0, Ok(())));
    }

    /// Without scaling the table is plain RoPE, and the first pair is exactly `base^-0 = 1`.
    #[test]
    fn plain_matches_the_base() {
        let f = rope_freqs_of(1e7, 64, None);
        assert_eq!(f.len(), 32);
        assert!((f[0] - 1.0).abs() < 1e-6);
        let want = 1e7f64.powf(-2.0 / 64.0);
        assert!((f[1] as f64 - want).abs() < 1e-6 * want);
    }

    /// YaRN leaves the short wavelengths alone, divides the long ones by the factor, and never goes up.
    #[test]
    fn yarn_interpolates_the_long_wavelengths() {
        let y = Yarn { factor: 4.0, orig_ctx: 262144, beta_fast: 32.0, beta_slow: 1.0 };
        let plain = rope_freqs_of(1e7, 64, None);
        let yarn = rope_freqs_of(1e7, 64, Some(y));
        assert!((yarn[0] - plain[0]).abs() < 1e-9, "pair 0: {} vs {}", yarn[0], plain[0]);
        let (last, p) = (yarn[31], plain[31]);
        assert!((last as f64 / p as f64 - 0.25).abs() < 0.01, "pair 31: {last} vs {p}");
        assert!(yarn.windows(2).all(|w| w[0] > w[1]), "frequencies must fall with the pair index");
    }

    /// A factor of 1 changes nothing, however the ramp is shaped.
    #[test]
    fn factor_one_is_the_identity() {
        let y = Yarn { factor: 1.0, orig_ctx: 131072, beta_fast: 32.0, beta_slow: 1.0 };
        let plain = rope_freqs_of(1e7, 64, None);
        let scaled = rope_freqs_of(1e7, 64, Some(y));
        for (a, b) in plain.iter().zip(&scaled) { assert!((a - b).abs() < 1e-9); }
    }
}
