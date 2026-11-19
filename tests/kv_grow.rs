//! The KV cache starts at `--ctx-start` and grows towards `ctx` when a conversation needs it. This lives in its own
//! test binary because it pins the expert placement through the environment, and `cargo test` runs the tests of one
//! binary in the same process: the settings would leak into the others.
mod common;
use rusting_strata::{gguf::Gguf, gpu_model::GpuModel, model::Model, ops, tokenizer::Tokenizer};

#[test]
fn kv_cache_grows_on_demand() {
    let _gpu = common::gpu_lock();
    let Some(p) = common::model_path() else { return };
    // Same experts in the same slots whatever the budget: only the KV allocation differs between the two runs.
    unsafe {
        std::env::set_var("STRATA_FIXED_EXPERTS", "1");
        std::env::set_var("STRATA_EXPERT_MB", "512");
    }
    let g = Gguf::open(&p).unwrap();
    let m = Model::load(&g).unwrap();
    let tok = Tokenizer::from_gguf(&g).unwrap();
    let ids = tok.encode(&"The river town woke before dawn. Fishermen checked their nets. ".repeat(40)).unwrap();
    assert!(ids.len() > 300, "need a prompt past the starting window, got {}", ids.len());

    let run = |kv_start: usize| -> (Vec<u32>, usize) {
        let gpu = Box::new(rusting_strata::cuda::Gpu::new(0).unwrap());
        let mut gm = GpuModel::load_with(&m, gpu, 1024, 128, 512 << 20, 0, kv_start).unwrap();
        let mut logits = gm.prefill(&ids).unwrap();
        let mut out = Vec::new();
        for _ in 0..8 {
            let t = ops::argmax(&logits) as u32;
            out.push(t);
            logits = gm.step(t).unwrap();
        }
        (out, gm.kv_cap)
    };

    let (want, full) = run(1024);
    let (got, grown) = run(128);
    assert_eq!(full, 1024, "a run with room from the start never grows");
    assert!(grown > ids.len(), "the cache grew past the prompt: {grown} for {} tokens", ids.len());
    assert_eq!(got, want, "tokens after the cache grew");
}
