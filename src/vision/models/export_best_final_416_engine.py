#!/usr/bin/env python3
"""
Export 0615 final YOLO model to TensorRT engine.

Input:
  ./best_final.pt

Output:
  ./best_final.engine

Expected class names:
  0: car
  1: person
  2: tank
  3: rock
  4: house
"""

from pathlib import Path
import os
import sys


EXPECTED_NAMES = {
    0: "car",
    1: "person",
    2: "tank",
    3: "rock",
    4: "house",
}


def normalize_names(names):
    if isinstance(names, dict):
        return {int(k): str(v) for k, v in names.items()}
    return {i: str(v) for i, v in enumerate(names)}


def main() -> int:
    model_dir = Path(__file__).resolve().parent
    pt_path = model_dir / "best_final.pt"
    engine_path = model_dir / "best_final.engine"

    imgsz = 416
    batch = 1
    device = "0"

    if not pt_path.exists():
        print(f"[ERROR] PT model not found: {pt_path}")
        return 1

    try:
        import torch
        print(f"[INFO] torch: {torch.__version__}")
        print(f"[INFO] torch.cuda.is_available(): {torch.cuda.is_available()}")

        if not torch.cuda.is_available():
            print("[ERROR] CUDA is not available. TensorRT engine export requires NVIDIA GPU.")
            return 1

        print(f"[INFO] CUDA device count: {torch.cuda.device_count()}")
        print(f"[INFO] CUDA device 0: {torch.cuda.get_device_name(0)}")

    except Exception as e:
        print(f"[ERROR] Failed to check torch/CUDA: {repr(e)}")
        return 1

    try:
        from ultralytics import YOLO
    except Exception as e:
        print(f"[ERROR] Failed to import ultralytics YOLO: {repr(e)}")
        return 1

    os.environ.setdefault("CUDA_MODULE_LOADING", "LAZY")

    print(f"[INFO] Loading PT model: {pt_path}")
    model = YOLO(str(pt_path))

    model_names = normalize_names(model.names)
    print(f"[INFO] Model names: {model_names}")

    if model_names != EXPECTED_NAMES:
        print("[ERROR] best_final.pt class names do not match 0615 final model.")
        print(f"        expected: {EXPECTED_NAMES}")
        print(f"        actual  : {model_names}")
        print("[ERROR] Wrong .pt file may be in this directory. Export aborted.")
        return 1

    print("[OK] Class names match 0615 final model.")
    print("[INFO] 0615 final model metrics expected:")
    print("       Precision 0.980 / Recall 0.954 / mAP50 0.976 / mAP50-95 0.883")
    print("       classes: car, person, tank, rock, house")

    if engine_path.exists():
        print(f"[INFO] Remove existing engine: {engine_path}")
        engine_path.unlink()

    print("[INFO] Export TensorRT engine")
    print(f"       input : {pt_path}")
    print(f"       output: {engine_path}")
    print(f"       imgsz : {imgsz}")
    print(f"       batch : {batch}")
    print(f"       device: {device}")
    print("       fp16  : true")
    print("       dynamic: false")

    try:
        exported = model.export(
            format="engine",
            imgsz=imgsz,
            batch=batch,
            device=device,
            half=True,
            dynamic=False,
            simplify=True,
            nms=False,
            workspace=4,
            verbose=True,
        )
    except TypeError:
        exported = model.export(
            format="engine",
            imgsz=imgsz,
            batch=batch,
            device=device,
            half=True,
            dynamic=False,
            simplify=True,
            nms=False,
            verbose=True,
        )
    except Exception as e:
        print(f"[ERROR] TensorRT export failed: {repr(e)}")
        return 1

    exported_path = Path(exported).resolve()
    print(f"[INFO] Exported path from ultralytics: {exported_path}")

    if exported_path != engine_path.resolve():
        if engine_path.exists():
            engine_path.unlink()
        exported_path.rename(engine_path)

    if not engine_path.exists():
        print(f"[ERROR] Engine file was not created: {engine_path}")
        return 1

    size_mb = engine_path.stat().st_size / (1024 * 1024)
    print("[OK] TensorRT engine created")
    print(f"     {engine_path}")
    print(f"     size: {size_mb:.2f} MB")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
