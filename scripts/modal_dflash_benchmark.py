"""
DFlash inference benchmark on Modal GPU platform.

Loads a pre-trained DFlash draft model from HuggingFace and benchmarks
speculative decoding against a target-only baseline.

Setup (one-time — same secrets as modal_dflash_train.py):
    modal token set --token-id <id> --token-secret <secret>
    modal secret create huggingface-secret HF_TOKEN=hf_...

Usage:
    # Default: benchmark Xingh3/dflash-qwen3-8b-1epoch (10 prompts, 128 tokens)
    modal run scripts/modal_dflash_benchmark.py

    # Custom HF draft model
    modal run scripts/modal_dflash_benchmark.py \
        --draft-repo Xingh3/dflash-qwen3-8b-1epoch

    # More prompts, longer generation
    modal run scripts/modal_dflash_benchmark.py \
        --num-prompts 50 --max-new-tokens 256

    # Skip baseline (only measure DFlash)
    modal run scripts/modal_dflash_benchmark.py --skip-baseline

    # Custom target model
    modal run scripts/modal_dflash_benchmark.py --target-model Qwen/Qwen3-8B

    # Custom block size and temperature
    modal run scripts/modal_dflash_benchmark.py --block-size 8 --temperature 0.6
"""

from __future__ import annotations

from typing import Optional

import modal

# =============================================================================
# Constants
# =============================================================================

TORCHSPEC_REPO = "https://github.com/zhubohao911/TorchSpec.git"
TORCHSPEC_BRANCH = "feature/dflash-training"

REPO_DIR = "/workspace/TorchSpec"
HF_CACHE_DIR = "/root/.cache/huggingface"

BENCHMARK_GPU = "H100:1"

# =============================================================================
# Modal app + volumes
# =============================================================================

app = modal.App("torchspec-dflash-benchmark")

hf_cache_vol = modal.Volume.from_name("hf-cache", create_if_missing=True)

# =============================================================================
# Container image — lightweight: just PyTorch + Transformers + TorchSpec
# =============================================================================

benchmark_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.0-devel-ubuntu22.04", add_python="3.11"
    )
    .apt_install("git")
    .pip_install(
        "torch", "torchvision", "torchaudio",
        extra_index_url="https://download.pytorch.org/whl/cu124",
    )
    .run_commands(
        f"git clone {TORCHSPEC_REPO} {REPO_DIR}",
        f"cd {REPO_DIR} && git checkout {TORCHSPEC_BRANCH}",
    )
    .pip_install(
        "huggingface_hub[hf_transfer]",
        "hf_transfer",
        "transformers==4.57.1",
        "datasets",
        "accelerate",
        "safetensors",
    )
    .run_commands(f"cd {REPO_DIR} && pip install -e '.[dev]'")
    .env(
        {
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            "PYTORCH_ALLOC_CONF": "expandable_segments:True",
            "HF_HOME": HF_CACHE_DIR,
        }
    )
)

# =============================================================================
# Benchmark runner — executed inside Modal container
# =============================================================================

@app.function(
    image=benchmark_image,
    gpu=BENCHMARK_GPU,
    volumes={HF_CACHE_DIR: hf_cache_vol},
    timeout=2 * 3600,
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def run_benchmark(
    target_model: str,
    draft_repo: str,
    num_prompts: int,
    max_new_tokens: int,
    block_size: int,
    temperature: float,
    skip_baseline: bool,
):
    import json
    import math
    import os
    import time
    from typing import List, Tuple

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

    from torchspec.models.draft.dflash import (
        DFlashConfig,
        DFlashDraftModel,
        build_target_layer_ids,
    )

    device = "cuda"
    print(f"Device: {device}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"VRAM: {mem_gb:.1f} GB")

    # ── Load target model ──
    print(f"\nLoading target model: {target_model}...")
    tokenizer = AutoTokenizer.from_pretrained(target_model, trust_remote_code=True)
    target = AutoModelForCausalLM.from_pretrained(
        target_model,
        torch_dtype=torch.bfloat16,
        device_map=device,
        trust_remote_code=True,
    )
    target.eval()
    target_params = sum(p.numel() for p in target.parameters()) / 1e9
    print(f"Target model loaded ({target_params:.1f}B params)")

    # ── Load draft model from HuggingFace ──
    print(f"\nLoading DFlash draft model: {draft_repo}...")
    draft = DFlashDraftModel.from_pretrained(
        draft_repo,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    ).to(device)
    draft.eval()

    target_layer_ids = draft.target_layer_ids
    context_proj = draft.context_proj
    context_norm = draft.context_norm

    draft_params = sum(p.numel() for p in draft.parameters()) / 1e6
    trainable_params = sum(p.numel() for p in draft.parameters() if p.requires_grad) / 1e6
    print(f"Draft model loaded ({draft_params:.1f}M total, {trainable_params:.1f}M trainable)")
    print(f"Target layer IDs: {target_layer_ids}")
    print(f"Block size: {block_size}")

    # ── Helpers (inlined from benchmark_dflash_inference.py) ──

    def extract_context_feature(
        hidden_states: list[torch.Tensor],
        layer_ids: list[int],
    ) -> torch.Tensor:
        offset = 1
        selected = [hidden_states[lid + offset] for lid in layer_ids]
        return torch.cat(selected, dim=-1)

    def sample(logits: torch.Tensor, temp: float = 0.0) -> torch.Tensor:
        if temp < 1e-5:
            return torch.argmax(logits, dim=-1)
        logits = logits / temp
        probs = torch.softmax(logits, dim=-1)
        if logits.dim() == 3:
            bsz, seq_len, vocab = logits.shape
            return torch.multinomial(
                probs.view(-1, vocab), num_samples=1
            ).view(bsz, seq_len)
        return torch.multinomial(probs, num_samples=1).squeeze(-1)

    def has_stop_token(token_ids: torch.Tensor) -> bool:
        eos_id = tokenizer.eos_token_id
        if eos_id is None:
            return False
        return (token_ids == eos_id).any().item()

    @torch.inference_mode()
    def generate_baseline(
        input_ids: torch.LongTensor,
        max_new: int,
    ) -> Tuple[torch.Tensor, float]:
        t0 = time.perf_counter()
        past_key_values = DynamicCache()
        generated = input_ids.clone()

        out = target(input_ids, past_key_values=past_key_values, use_cache=True)
        next_token = sample(out.logits[:, -1:, :], temperature)
        generated = torch.cat([generated, next_token], dim=1)

        for _ in range(max_new - 1):
            out = target(next_token, past_key_values=past_key_values, use_cache=True)
            next_token = sample(out.logits[:, -1:, :], temperature)
            generated = torch.cat([generated, next_token], dim=1)
            if has_stop_token(next_token):
                break

        elapsed = time.perf_counter() - t0
        return generated, elapsed

    @torch.inference_mode()
    def generate_dflash_spec(
        input_ids: torch.LongTensor,
        max_new: int,
    ) -> Tuple[torch.Tensor, float, List[int]]:
        num_input = input_ids.shape[1]
        max_length = num_input + max_new
        mask_token_id = draft.mask_token_id

        output_ids = torch.full(
            (1, max_length + block_size),
            mask_token_id,
            dtype=torch.long,
            device=device,
        )
        output_ids[:, :num_input] = input_ids

        t0 = time.perf_counter()

        past_kv_target = DynamicCache()
        out = target(
            input_ids,
            past_key_values=past_kv_target,
            use_cache=True,
            output_hidden_states=True,
        )

        first_token = sample(out.logits[:, -1:, :], temperature)
        output_ids[:, num_input] = first_token.squeeze()

        ctx_hidden = extract_context_feature(out.hidden_states, target_layer_ids)
        ctx_feature = context_norm(context_proj(ctx_hidden.to(context_proj.weight.dtype)))

        acceptance_lengths = []
        start = num_input

        while start < max_length:
            block_ids = output_ids[:, start : start + block_size].clone()

            draft_pos = torch.arange(
                start, start + block_size, device=device
            ).unsqueeze(0)
            ctx_pos = torch.arange(start, device=device).unsqueeze(0)

            draft_hidden = draft(
                draft_input_ids=block_ids,
                context_feature=ctx_feature[:, :start, :],
                draft_position_ids=draft_pos,
                context_position_ids=ctx_pos,
                block_mask=None,
            )

            draft_logits = target.lm_head(draft_hidden[:, 1:, :])
            draft_tokens = sample(draft_logits, temperature)

            block_ids[:, 1:] = draft_tokens

            out_verify = target(
                block_ids,
                position_ids=draft_pos,
                past_key_values=past_kv_target,
                use_cache=True,
                output_hidden_states=True,
            )

            posterior = sample(out_verify.logits, temperature)

            match = (block_ids[:, 1:] == posterior[:, :-1]).cumprod(dim=1)
            acc_len = match.sum(dim=1)[0].item()

            output_ids[:, start : start + acc_len + 1] = block_ids[:, : acc_len + 1]
            output_ids[:, start + acc_len + 1] = posterior[:, acc_len]

            start += acc_len + 1
            past_kv_target.crop(start)

            ctx_hidden_new = extract_context_feature(
                out_verify.hidden_states, target_layer_ids
            )
            new_feat = context_norm(
                context_proj(ctx_hidden_new.to(context_proj.weight.dtype))
            )
            ctx_feature = torch.cat(
                [ctx_feature[:, :start - acc_len - 1, :], new_feat[:, : acc_len + 1, :]],
                dim=1,
            )

            acceptance_lengths.append(acc_len + 1)

            if has_stop_token(output_ids[:, start - 1 : start]):
                break

        elapsed = time.perf_counter() - t0

        output_ids = output_ids[:, :max_length]
        valid_mask = output_ids[0] != mask_token_id
        output_ids = output_ids[:, valid_mask]

        return output_ids, elapsed, acceptance_lengths

    # ── Prompts ──
    prompts = [
        "Explain the theory of relativity in simple terms.",
        "What is quantum entanglement?",
        "How does nuclear fusion work and why is it hard to achieve?",
        "Describe the Big Bang theory and the evidence supporting it.",
        "What are gravitational waves and how were they first detected?",
        "Write a Python function that implements binary search.",
        "Describe how a transformer neural network works.",
        "What is the difference between TCP and UDP?",
        "Explain how a compiler works step by step.",
        "What are the key principles of object-oriented programming?",
        "How does garbage collection work in modern programming languages?",
        "Explain the CAP theorem in distributed systems.",
        "How do vaccines work to protect against diseases?",
        "How does the human immune system fight infections?",
        "Explain the process of DNA replication in cells.",
        "What is CRISPR and how does it edit genes?",
        "Describe the stages of mitosis and meiosis.",
        "What is blockchain technology and how does it work?",
        "What are the benefits and risks of artificial intelligence?",
        "How do self-driving cars perceive their environment?",
        "Explain how 5G networks differ from 4G.",
        "What is edge computing and when should it be used?",
        "Explain the Monty Hall problem and its solution.",
        "What is Bayes' theorem and how is it applied?",
        "Describe the P vs NP problem in computer science.",
        "What is the traveling salesman problem?",
        "What are the main causes of climate change?",
        "Explain the process of making bread from scratch.",
        "Describe the solar system and its planets.",
        "How does a microwave oven heat food?",
        "Why is the sky blue during the day and red at sunset?",
        "Describe the key events of the French Revolution.",
        "What caused the fall of the Roman Empire?",
        "Explain the significance of the printing press.",
        "How did the Industrial Revolution change society?",
        "What were the main causes of World War I?",
        "How does a jet engine produce thrust?",
        "Explain how bridges are designed to handle stress.",
        "What is the difference between AC and DC electricity?",
        "How do solar panels convert sunlight to electricity?",
        "Describe how a lithium-ion battery works.",
        "What is the trolley problem in ethics?",
        "Explain the concept of supply and demand in economics.",
        "What is cognitive dissonance in psychology?",
        "Describe the scientific method step by step.",
        "What is game theory and how is it applied?",
        "Write a short story about a robot learning to paint.",
        "Explain the rules of haiku poetry with examples.",
        "What makes a good persuasive essay?",
        "How does the internet work from a technical perspective?",
    ][:num_prompts]

    print(f"\n{'='*60}")
    print(f"Benchmark: {len(prompts)} prompts, max_new_tokens={max_new_tokens}")
    print(f"Draft model: {draft_repo}")
    print(f"Target model: {target_model}")
    print(f"Temperature: {temperature}, Block size: {block_size}")
    print(f"{'='*60}\n")

    # ── Baseline: target-only ──
    baseline_times = []
    baseline_tokens = []
    if not skip_baseline:
        print("Running BASELINE (target-only autoregressive)...")
        warmup_ids = tokenizer("Hello", return_tensors="pt")["input_ids"].to(device)
        generate_baseline(warmup_ids, 16)
        torch.cuda.synchronize()

        for i, prompt in enumerate(prompts):
            input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)
            torch.cuda.synchronize()
            output, elapsed = generate_baseline(input_ids, max_new_tokens)
            torch.cuda.synchronize()
            num_new = output.shape[1] - input_ids.shape[1]
            baseline_times.append(elapsed)
            baseline_tokens.append(num_new)
            if (i + 1) % 5 == 0:
                print(f"  [{i+1}/{len(prompts)}] {num_new} tokens in {elapsed:.2f}s "
                      f"({num_new/elapsed:.1f} tok/s)")

        avg_baseline_time = sum(baseline_times) / len(baseline_times)
        avg_baseline_tokens = sum(baseline_tokens) / len(baseline_tokens)
        avg_baseline_tps = sum(
            t / e for t, e in zip(baseline_tokens, baseline_times)
        ) / len(prompts)
        print(f"\n  Baseline avg: {avg_baseline_tokens:.0f} tokens, "
              f"{avg_baseline_time:.2f}s, {avg_baseline_tps:.1f} tok/s\n")

    # ── DFlash speculative decoding ──
    print("Running DFLASH speculative decoding...")
    warmup_ids = tokenizer("Hello", return_tensors="pt")["input_ids"].to(device)
    generate_dflash_spec(warmup_ids, 16)
    torch.cuda.synchronize()

    dflash_times = []
    dflash_tokens = []
    all_acc_lens = []

    for i, prompt in enumerate(prompts):
        input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)
        torch.cuda.synchronize()
        output, elapsed, acc_lens = generate_dflash_spec(input_ids, max_new_tokens)
        torch.cuda.synchronize()
        num_new = output.shape[1] - input_ids.shape[1]
        dflash_times.append(elapsed)
        dflash_tokens.append(num_new)
        all_acc_lens.extend(acc_lens)
        avg_tau = sum(acc_lens) / max(len(acc_lens), 1)
        if (i + 1) % 5 == 0:
            print(f"  [{i+1}/{len(prompts)}] {num_new} tokens in {elapsed:.2f}s "
                  f"({num_new/elapsed:.1f} tok/s, τ={avg_tau:.2f})")

    avg_dflash_time = sum(dflash_times) / len(dflash_times)
    avg_dflash_tokens = sum(dflash_tokens) / len(dflash_tokens)
    avg_dflash_tps = sum(
        t / e for t, e in zip(dflash_tokens, dflash_times)
    ) / len(prompts)
    avg_tau = sum(all_acc_lens) / max(len(all_acc_lens), 1)

    print(f"\n  DFlash avg: {avg_dflash_tokens:.0f} tokens, "
          f"{avg_dflash_time:.2f}s, {avg_dflash_tps:.1f} tok/s, τ={avg_tau:.2f}\n")

    # ── Per-prompt breakdown ──
    print(f"\n{'='*60}")
    print("PER-PROMPT BREAKDOWN")
    print(f"{'='*60}")
    for i, prompt in enumerate(prompts):
        tok_s = dflash_tokens[i] / dflash_times[i] if dflash_times[i] > 0 else 0
        print(f"  [{i+1:2d}] {dflash_tokens[i]:3d} tok, {dflash_times[i]:.2f}s, "
              f"{tok_s:.1f} tok/s  | {prompt[:60]}")

    # ── τ distribution ──
    print(f"\n{'='*60}")
    print("τ DISTRIBUTION (per draft cycle)")
    print(f"{'='*60}")
    if all_acc_lens:
        tau_counts = {}
        for a in all_acc_lens:
            tau_counts[a] = tau_counts.get(a, 0) + 1
        for k in sorted(tau_counts.keys()):
            bar = "█" * int(tau_counts[k] / max(tau_counts.values()) * 40)
            print(f"  τ={k:2d}: {tau_counts[k]:4d} ({tau_counts[k]/len(all_acc_lens)*100:5.1f}%) {bar}")
        print(f"  Total cycles: {len(all_acc_lens)}, mean τ={avg_tau:.2f}, "
              f"median τ={sorted(all_acc_lens)[len(all_acc_lens)//2]}")

    # ── Summary ──
    print(f"\n{'='*60}")
    print("RESULTS SUMMARY")
    print(f"{'='*60}")
    print(f"  Draft model:  {draft_repo}")
    print(f"  Target model: {target_model}")
    print(f"  GPU:          {torch.cuda.get_device_name(0)}")
    print(f"  Prompts:      {len(prompts)}")
    print(f"  Max tokens:   {max_new_tokens}")
    if not skip_baseline:
        speedup = avg_baseline_time / avg_dflash_time if avg_dflash_time > 0 else 0
        print(f"  Baseline:     {avg_baseline_tps:.1f} tok/s")
        print(f"  DFlash:       {avg_dflash_tps:.1f} tok/s (τ={avg_tau:.2f})")
        print(f"  Speedup:      {speedup:.2f}x")
    else:
        print(f"  DFlash:       {avg_dflash_tps:.1f} tok/s (τ={avg_tau:.2f})")
    print(f"{'='*60}")


# =============================================================================
# CLI entry point
# =============================================================================

@app.local_entrypoint()
def main(
    target_model: str = "Qwen/Qwen3-8B",
    draft_repo: str = "Xingh3/dflash-qwen3-8b-1epoch",
    num_prompts: int = 10,
    max_new_tokens: int = 128,
    block_size: int = 16,
    temperature: float = 0.0,
    skip_baseline: bool = False,
):
    print("=" * 60)
    print("  DFlash Inference Benchmark on Modal")
    print("=" * 60)
    print(f"  Target model:  {target_model}")
    print(f"  Draft model:   {draft_repo}")
    print(f"  GPU:           {BENCHMARK_GPU}")
    print(f"  Num prompts:   {num_prompts}")
    print(f"  Max new tokens:{max_new_tokens}")
    print(f"  Block size:    {block_size}")
    print(f"  Temperature:   {temperature}")
    print(f"  Skip baseline: {skip_baseline}")
    print("=" * 60)

    run_benchmark.remote(
        target_model=target_model,
        draft_repo=draft_repo,
        num_prompts=num_prompts,
        max_new_tokens=max_new_tokens,
        block_size=block_size,
        temperature=temperature,
        skip_baseline=skip_baseline,
    )
