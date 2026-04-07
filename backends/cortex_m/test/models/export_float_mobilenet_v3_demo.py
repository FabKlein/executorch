#!/usr/bin/env python3
# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import json
import random
from pathlib import Path

import executorch.backends.cortex_m.ops.operators  # noqa: F401

import numpy as np
import torch
from datasets import load_dataset
from executorch.backends.cortex_m.passes.cortex_m_pass_manager import CortexMPassManager
from executorch.exir import EdgeCompileConfig
from executorch.extension.export_util.utils import export_to_edge, save_pte_program
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights


def write_storage_bytes(path: Path, tensor: torch.Tensor) -> None:
    path.write_bytes(bytes(tensor.untyped_storage()))


def load_sample(
    sample_index: int | None, sample_seed: int
) -> tuple[torch.Tensor, str, int]:
    dataset = load_dataset("frgfm/imagenette", "full_size", split="validation")
    chosen_index = sample_index
    if chosen_index is None:
        chosen_index = random.Random(sample_seed).randrange(len(dataset))
    sample = dataset[int(chosen_index)]

    weights = MobileNet_V3_Small_Weights.DEFAULT
    preprocess = weights.transforms()
    label_names = dataset.features["label"].names

    image = sample["image"].convert("RGB")
    transformed = preprocess(image).unsqueeze(0).to(memory_format=torch.channels_last)
    return transformed, label_names[sample["label"]], int(chosen_index)


def top1_label(output: torch.Tensor) -> str:
    categories = MobileNet_V3_Small_Weights.DEFAULT.meta["categories"]
    return categories[int(output.argmax().item())]


def print_topk(output: torch.Tensor, k: int = 10) -> None:
    categories = MobileNet_V3_Small_Weights.DEFAULT.meta["categories"]
    probs = torch.softmax(output[0].to(torch.float32), dim=0)
    values, indices = torch.topk(probs, k)
    print(f"Top-{k} probabilities:")
    for rank, (value, index) in enumerate(
        zip(values.tolist(), indices.tolist()), start=1
    ):
        print(f"  {rank:2d}. {categories[index]}: {value:.6f}")


def build_model(dtype: torch.dtype) -> torch.nn.Module:
    model = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT).eval()
    if dtype == torch.float16:
        model = model.half()
    else:
        model = model.float()
    return model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--output_dir", default=".")
    parser.add_argument("--dtype", choices=("float32", "float16"), default="float32")
    parser.add_argument(
        "--mode",
        choices=("pytorch", "prepare"),
        default="prepare",
        help="Use 'pytorch' to run only host-side inference, or 'prepare' to also export a .pte and artifacts.",
    )
    parser.add_argument(
        "--sample_index",
        type=int,
        default=None,
        help="Validation-split sample index. If omitted, a deterministic random index is selected from --sample_seed.",
    )
    parser.add_argument(
        "--sample_seed",
        type=int,
        default=0,
        help="Seed used to choose a random validation sample when --sample_index is omitted.",
    )
    parser.add_argument(
        "--dump_output",
        action="store_true",
        help="Write raw model output to output.npy and output.bin in --output_dir.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dtype = torch.float32 if args.dtype == "float32" else torch.float16
    sample_tensor, true_label, chosen_index = load_sample(
        args.sample_index, args.sample_seed
    )
    sample_tensor = sample_tensor.to(dtype)

    model = build_model(dtype)
    with torch.no_grad():
        output = model(sample_tensor)
    predicted_label = top1_label(output)

    meta = {
        "dtype": args.dtype,
        "true_label": true_label,
        "predicted_label": predicted_label,
        "sample_index": chosen_index,
    }

    print(f"Sample index   : {chosen_index}")
    print(f"True label     : {true_label}")
    print(f"PyTorch top-1  : {predicted_label}")
    print_topk(output, k=10)

    if args.dump_output:
        output_np = output.detach().cpu().to(torch.float32).numpy()
        np.save(output_dir / "output.npy", output_np)
        (output_dir / "output.bin").write_bytes(output.detach().cpu().numpy().tobytes())
        (output_dir / "run_meta.json").write_text(json.dumps(meta, indent=2))

    if args.mode == "pytorch":
        return

    edge_program = export_to_edge(
        model,
        (sample_tensor,),
        edge_compile_config=EdgeCompileConfig(_check_ir_validity=False),
    )
    edge_program._edge_programs["forward"] = CortexMPassManager(
        edge_program.exported_program()
    ).transform()
    program = edge_program.to_executorch()

    suffix = "f32" if dtype == torch.float32 else "f16"
    pte_base = f"mobilenet_v3_small_{suffix}_demo"
    save_pte_program(program, pte_base, str(output_dir))
    write_storage_bytes(output_dir / "i0.bin", sample_tensor)
    write_storage_bytes(output_dir / "expected-0.bin", output)
    (output_dir / "run_meta.json").write_text(json.dumps(meta, indent=2))


if __name__ == "__main__":
    with torch.no_grad():
        main()
