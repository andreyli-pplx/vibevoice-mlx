"""Fixed-duration, two-speaker generation benchmark (run each backend separately)."""

import argparse
import hashlib
import json
import logging
import resource
import subprocess
import time
from pathlib import Path

import mlx.core as mx
import numpy as np
import soundfile as sf

from vibevoice_mlx.e2e_pipeline import (
    _try_coreml_semantic,
    _try_mlx_semantic,
    encode_voice_reference,
    tokenize_text,
)
from vibevoice_mlx.generate import GenerationOptions, generate
from vibevoice_mlx.load_weights import load_model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--text-file", type=Path, required=True)
    parser.add_argument("--ref-audio", nargs=2, required=True)
    parser.add_argument("--backend", choices=["mlx", "coreml", "ane"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tokens", type=int, default=2250)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    start_setup = time.perf_counter()
    model, config = load_model(args.model, quantize_bits=8)
    text = args.text_file.read_text()
    prompt = tokenize_text(text, args.model, config, ref_audio=args.ref_audio)
    voice_embeds = {}
    for speaker in prompt.speakers:
        embeds = encode_voice_reference(
            speaker.ref_audio_np, speaker.num_vae_tokens, model, config, args.model
        )
        for i, pos in enumerate(speaker.speech_embed_positions):
            voice_embeds[pos] = mx.array(embeds[i : i + 1]).astype(mx.float16)
    semantic = (
        _try_mlx_semantic(model, config, args.model)
        if args.backend == "mlx"
        else _try_coreml_semantic(model, config, use_ane=args.backend == "ane")
    )
    if semantic is None:
        raise RuntimeError(
            f"Requested {args.backend} backend did not load; refusing fallback"
        )
    semantic_fn, semantic_reset = semantic
    setup_seconds = time.perf_counter() - start_setup
    start_warmup = time.perf_counter()
    generate(
        model,
        prompt.input_ids,
        GenerationOptions(
            diffusion_steps=10, cfg_scale=1.3, max_speech_tokens=8, seed=42
        ),
        semantic_encoder_fn=semantic_fn,
        semantic_reset_fn=semantic_reset,
        voice_embeds=voice_embeds,
        estimated_total=8,
    )
    semantic_reset()
    mx.synchronize()
    warmup_seconds = time.perf_counter() - start_warmup
    mx.reset_peak_memory()
    start = time.perf_counter()
    audio, metrics = generate(
        model,
        prompt.input_ids,
        GenerationOptions(
            diffusion_steps=10, cfg_scale=1.3, max_speech_tokens=args.tokens, seed=42
        ),
        semantic_encoder_fn=semantic_fn,
        semantic_reset_fn=semantic_reset,
        voice_embeds=voice_embeds,
        estimated_total=args.tokens,
    )
    mx.synchronize()
    seconds = time.perf_counter() - start
    duration = len(audio) / 24000
    report = {
        "backend": args.backend,
        "commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "source_diff": subprocess.check_output(
            ["git", "diff", "--", "vibevoice_mlx"], text=True
        ),
        "model": args.model,
        "quantization_bits": 8,
        "diffusion_steps": 10,
        "seed": 42,
        "cfg_scale": 1.3,
        "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "reference_sha256": [
            hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in args.ref_audio
        ],
        "setup_seconds": setup_seconds,
        "warmup_seconds": warmup_seconds,
        "generation_seconds": seconds,
        "audio_seconds": duration,
        "audio_seconds_per_wall_second": duration / seconds,
        "wall_seconds_per_audio_second": seconds / duration,
        "mlx_peak_memory_gib": mx.get_peak_memory() / 2**30,
        "process_peak_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        / 2**30,
        "finite": bool(np.isfinite(audio).all()),
        "peak_amplitude": float(np.max(np.abs(audio))),
        "out_of_pcm_range_samples": int(np.count_nonzero(np.abs(audio) >= 1)),
        "metrics": metrics.summary(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(args.output), audio, 24000, subtype="PCM_16")
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)
    if len(audio) != args.tokens * 3200:
        raise RuntimeError(
            f"Generated {duration:.2f}s, expected {args.tokens * 3200 / 24000:.2f}s"
        )
    if not report["finite"]:
        raise RuntimeError("Non-finite audio")


if __name__ == "__main__":
    main()
