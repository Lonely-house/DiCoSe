"""Cache frozen real CLAP text embeddings for manifest prompts."""

import argparse
import json
from pathlib import Path

import torch

from arb_stem_mss.text import collect_prompts, encode_texts, load_embedding_cache, mock_cache


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifests", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--clap-checkpoint")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--audio-model", default="HTSAT-base")
    parser.add_argument("--mock", action="store_true", help="Numerical test fixture only, without text semantics")
    parser.add_argument("--query-dim", type=int, default=512, help="Only controls explicit mock fixtures")
    parser.add_argument("--seed", type=int, default=42, help="Only controls explicit mock fixtures")
    args = parser.parse_args()
    if args.mock and args.clap_checkpoint:
        parser.error("--mock and --clap-checkpoint cannot be combined")
    if not args.mock and not args.clap_checkpoint:
        parser.error("Real text encoding requires --clap-checkpoint; --mock is only for smoke tests")
    destination = Path(args.output)
    if destination.exists():
        parser.error("Output already exists; choose a new filename to preserve encoder provenance")
    prompts, manifest_hashes = collect_prompts(args.manifests)
    if args.mock:
        cache = mock_cache(prompts, args.query_dim, args.seed)
    else:
        cache = encode_texts(prompts, args.clap_checkpoint, args.device, args.batch_size, args.audio_model)
    cache["manifest_sha256"] = manifest_hashes
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(cache, destination)
    load_embedding_cache(destination, allow_mock=args.mock)
    metadata = {key: value for key, value in cache.items() if key != "vectors"}
    metadata["prompt_count"] = len(cache["vectors"])
    metadata["prompts"] = list(cache["vectors"])
    destination.with_suffix(destination.suffix + ".json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(destination), "encoder_id": cache["encoder_id"],
                      "prompt_count": len(prompts), "query_dim": cache["query_dim"],
                      "is_mock": cache["is_mock"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
