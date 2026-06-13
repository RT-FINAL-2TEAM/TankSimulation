from pathlib import Path
import numpy as np
from ultralytics import YOLO

MODEL_DIR = Path("/home/tankcc/tankcc/src/vision/models")
PT_PATH = MODEL_DIR / "best_final.pt"
ENGINE_PATH = MODEL_DIR / "best_final.engine"

def main():
    if not PT_PATH.exists():
        raise FileNotFoundError(f"PT model not found: {PT_PATH}")

    print(f"[load] {PT_PATH}")
    model = YOLO(str(PT_PATH), task="detect")

    print("[export] TensorRT engine 생성 시작")
    print("[config] imgsz=416, half=True, device=0, dynamic=False")

    exported_path = model.export(
        format="engine",
        imgsz=416,
        half=True,
        device=0,
        dynamic=False,
        simplify=True,
        workspace=4,
        verbose=True,
    )

    exported_path = Path(exported_path)

    print(f"[exported] {exported_path}")

    if not ENGINE_PATH.exists():
        raise FileNotFoundError(f"Engine export failed: {ENGINE_PATH}")

    print("[test] 416 engine 로딩 테스트")
    trt_model = YOLO(str(ENGINE_PATH), task="detect")

    dummy = np.zeros((416, 416, 3), dtype=np.uint8)
    results = trt_model.predict(
        source=dummy,
        imgsz=416,
        device=0,
        half=True,
        verbose=False,
    )

    print(f"[ok] 416 TensorRT engine 생성 완료: {ENGINE_PATH}")
    print(f"[test] result count: {len(results)}")

if __name__ == "__main__":
    main()