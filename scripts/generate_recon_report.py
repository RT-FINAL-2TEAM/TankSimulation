#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""정찰 보고서 생성기 — recon_reports/*.json → A/B 위험도·은밀성 비교 + 루트 추천.

파이프라인상 위치 (ROS·시뮬 불필요, 순수 후처리):
    run_recon_scenario.py  ──(ROS 런타임)──▶  route_A.json / route_B.json / comparison.json
                                                          │
                                                          ▼
                          generate_recon_report.py  ──▶  recon_reports/recon_report.md

명세(Tank_System_Spec.pdf 6.2) 위험도 수식:
    Risk = W1·(적/초소 발견) + W2·(시야 노출시간) + W3·(우회/이탈) + W4·(지형굴곡 σP+σR) + W5·(소요시간)
  - W1: vision_yolo.counts 위협클래스(person/tank/house; rock/car 제외) 가중합 + asset_spotted_gt 보강
  - W2: finalmap.map GT 위협(초소/적전차) + 전차 궤적으로 사후 계산(노출시간·발각횟수)
  - W3: distance_m / 직선거리(start→goal) 우회비(재계획 카운트 미로깅이라 근사)
  - W4: terrain_roughness.pitch_std + roll_std
  - W5: result.sim_time_s
  값이 낮을수록 안전(은밀)한 루트. 추천은 유효 런 중 최저 위험.

사용:
    python3 scripts/generate_recon_report.py
    python3 scripts/generate_recon_report.py --input recon_reports --stdout
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from recon_eval import threat_geometry as tg  # noqa: E402

# --------------------------------------------------------------------------- #
# 상수
# --------------------------------------------------------------------------- #
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_REPORT_DIR = os.path.join(PROJECT_ROOT, "recon_reports")
DEFAULT_MAP = os.path.join(PROJECT_ROOT, "src", "rviz_visualization", "map", "finalmap.map")
DEFAULT_ROUTES = os.path.join(PROJECT_ROOT, "src", "path_planning", "config", "routes.yaml")

# 출발/목적 (routes.yaml finalmap 기준). 우회비 직선거리 계산에 사용.
START_XY = (60.0, 30.0)
GOAL_XY = (110.0, 276.5)

# 위험도 가중치 (은밀성 우선: 발견 W1 / 노출 W2 높게). 합 1.0.
DEFAULT_WEIGHTS = {"W1": 0.35, "W2": 0.30, "W3": 0.10, "W4": 0.15, "W5": 0.10}

# W1 프록시: YOLO 위협클래스별 가중. YOLO 클래스 = person/tank/rock/house/car.
# 위협(시야로 발각): person(병사)/tank(적전차)/house(초소). rock/car는 비위협 장애물 → 제외.
# (house는 초소로 FOV 보유. 더 위험하게 보려면 가중을 1.0↑로.)
THREAT_CLASS_WEIGHTS = {"person": 1.0, "tank": 1.0, "house": 1.0}

# reference 정규화 기준값 (해당 값이면 norm=1.0).
REFS = {
    "threat": 20.0,     # W1 위협 발견 가중합
    "exposure": 60.0,   # W2 시야 노출시간(s)
    "detour": 2.0,      # W3 우회비 상한(직선=1.0, 2.0=2배)
    "terrain": 8.0,     # W4 σPitch+σRoll(deg)
    "time": 300.0,      # W5 소요시간(s)
}

# 데이터 품질 임계
MIN_VALID_DISTANCE = 5.0   # m 미만 이동이면 무효 런(멈춤/미완주)
COLLISION_WARN = 10        # 초과 시 충돌 과다 경고
PERCEPTION_MATCH_TOL = 3.0  # m, LiDAR 탐지↔GT 객체 최근접 매칭 허용오차

# YOLO 클래스 → GT prefab 접두사 매핑(센서 정확도 비교용)
YOLO_TO_GT_PREFAB = {"person": "Human", "tank": "Tank", "rock": "Rock", "house": "House", "car": "Car"}

EXIT_OK = 0
EXIT_NO_INPUT = 2
EXIT_NO_VALID_RUN = 3
EXIT_SINGLE_ROUTE = 4


# --------------------------------------------------------------------------- #
# 입력 로드
# --------------------------------------------------------------------------- #
def load_inputs(input_path: str, route_a: Optional[str], route_b: Optional[str]) -> Dict[str, dict]:
    """comparison.json 우선, 없으면 route_A/B.json 개별 로드. {'A':..,'B':..}."""
    routes: Dict[str, dict] = {}

    if route_a or route_b:
        for rid, path in (("A", route_a), ("B", route_b)):
            if path and os.path.isfile(path):
                routes[rid] = _read_json(path)
        return routes

    # input_path: 디렉터리 또는 comparison.json 파일
    comparison = input_path if input_path.endswith(".json") else os.path.join(input_path, "comparison.json")
    if os.path.isfile(comparison):
        data = _read_json(comparison)
        for key, rid in (("route_A", "A"), ("route_B", "B")):
            r = data.get(key)
            if isinstance(r, dict) and r:
                routes[rid] = r
        if routes:
            return routes

    # 폴백: route_A.json / route_B.json
    base = input_path if os.path.isdir(input_path) else os.path.dirname(input_path)
    for rid in ("A", "B"):
        p = os.path.join(base, f"route_{rid}.json")
        if os.path.isfile(p):
            routes[rid] = _read_json(p)
    return routes


def _read_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# --------------------------------------------------------------------------- #
# 지표 추출
# --------------------------------------------------------------------------- #
def compute_threat_proxy(yolo_counts: Dict[str, int]) -> Tuple[float, Dict[str, float]]:
    """vision_yolo.counts → 위협클래스 가중합 + 항목별 분해."""
    breakdown: Dict[str, float] = {}
    total = 0.0
    for cls, w in THREAT_CLASS_WEIGHTS.items():
        n = int(yolo_counts.get(cls, 0))
        if n:
            breakdown[cls] = round(n * w, 2)
            total += n * w
    return round(total, 2), breakdown


def _traj_points(report: dict) -> Optional[List[Tuple[float, float, float]]]:
    """report['trajectory'] → [(t,x,z)] 또는 None. dict/list 양식 모두 허용."""
    traj = report.get("trajectory")
    if not isinstance(traj, list) or len(traj) < 2:
        return None
    pts: List[Tuple[float, float, float]] = []
    for item in traj:
        try:
            if isinstance(item, dict):
                pts.append((float(item["t"]), float(item["x"]), float(item["z"])))
            elif isinstance(item, (list, tuple)) and len(item) >= 3:
                pts.append((float(item[0]), float(item[1]), float(item[2])))
        except (KeyError, TypeError, ValueError):
            continue
    return pts if len(pts) >= 2 else None


def extract_metrics(report: dict) -> dict:
    result = report.get("result", {})
    obstacle = report.get("obstacle_summary", {})
    terrain = report.get("terrain_roughness", {})
    yolo_counts = report.get("vision_yolo", {}).get("counts", {}) or {}

    distance = float(result.get("distance_m", 0.0))
    obstacle_count = int(obstacle.get("count", 0))
    # density는 원본의 0-division 잔재가 있어 직접 재계산(거리 유효할 때만).
    density = round(obstacle_count / (distance / 100.0), 2) if distance >= MIN_VALID_DISTANCE else None

    threat_proxy, threat_breakdown = compute_threat_proxy(yolo_counts)
    pitch = float(terrain.get("pitch_std_deg", 0.0))
    roll = float(terrain.get("roll_std_deg", 0.0))

    return {
        "route": str(report.get("route", "?")),
        "map": str(report.get("map", "")),
        "reached": bool(result.get("reached", False)),
        "distance_m": round(distance, 2),
        "sim_time_s": round(float(result.get("sim_time_s", 0.0)), 2),
        "collisions": int(result.get("collisions", 0)),
        "obstacle_count": obstacle_count,
        "obstacle_density": density,
        "terrain_sigma": round(pitch + roll, 4),
        "terrain_pitch": pitch,
        "terrain_roll": roll,
        "yolo_counts": dict(yolo_counts),
        "threat_proxy": threat_proxy,
        "threat_breakdown": threat_breakdown,
        "asset_gt": report.get("asset_spotted_gt", {}) or {},
        "detour_ratio": _detour_ratio(distance),
        "obstacles_detected": report.get("obstacles_detected", []) or [],
        "trajectory": _traj_points(report),
    }


def _detour_ratio(distance: float) -> Optional[float]:
    straight = math.hypot(GOAL_XY[0] - START_XY[0], GOAL_XY[1] - START_XY[1])
    if straight <= 1e-6 or distance < MIN_VALID_DISTANCE:
        return None
    return round(distance / straight, 3)


# --------------------------------------------------------------------------- #
# 노출/발각 사후 계산 (전차 궤적 + GT 위협)
# --------------------------------------------------------------------------- #
def compute_exposure(trajectory: Optional[List[Tuple[float, float, float]]],
                     threats: List[Dict[str, Any]],
                     gt_bboxes: List[Dict[str, float]]) -> Optional[dict]:
    """궤적 각 샘플에 is_threat_active 적용 → 노출시간/발각횟수/최소거리.

    반환 None = 궤적 미수집(N/A). 위협 0개면 0 노출로 반환.
    """
    if not trajectory:
        return None

    per_threat: List[dict] = []
    total_fov_dwell = 0.0
    total_prox_dwell = 0.0
    total_detections = 0
    global_max_continuous = 0.0

    for ti, threat in enumerate(threats):
        label = threat.get("prefabName") or f"{threat.get('type','threat')}#{ti}"
        radius = tg.threat_radius(threat)
        fov_dwell = 0.0
        prox_dwell = 0.0
        detections = 0
        max_continuous = 0.0
        cur_continuous = 0.0
        min_dist = float("inf")
        prev_active = False

        for i in range(1, len(trajectory)):
            t0, x0, z0 = trajectory[i - 1]
            t1, x1, z1 = trajectory[i]
            dt = max(0.0, t1 - t0)

            active = tg.is_threat_active((x1, z1), threat, gt_bboxes)
            dist = math.hypot(x1 - float(threat["x"]), z1 - float(threat["z"]))
            within_radius = dist <= radius

            if active:
                fov_dwell += dt
                if dist < min_dist:
                    min_dist = dist
                if not prev_active:
                    detections += 1          # 발각 진입 에지
                    cur_continuous = 0.0
                cur_continuous += dt
                max_continuous = max(max_continuous, cur_continuous)
            else:
                cur_continuous = 0.0
            if within_radius:
                prox_dwell += dt
            prev_active = active

        if fov_dwell > 0 or prox_dwell > 0:
            per_threat.append({
                "threat": label,
                "type": threat.get("type", "unknown"),
                "detections": detections,
                "fov_dwell_s": round(fov_dwell, 2),
                "proximity_dwell_s": round(prox_dwell, 2),
                "max_continuous_s": round(max_continuous, 2),
                "min_dist_m": round(min_dist, 2) if min_dist != float("inf") else None,
            })
        total_fov_dwell += fov_dwell
        total_prox_dwell += prox_dwell
        total_detections += detections
        global_max_continuous = max(global_max_continuous, max_continuous)

    per_threat.sort(key=lambda e: e["fov_dwell_s"], reverse=True)
    return {
        "total_fov_dwell_s": round(total_fov_dwell, 2),
        "total_proximity_dwell_s": round(total_prox_dwell, 2),
        "detection_count": total_detections,
        "max_continuous_s": round(global_max_continuous, 2),
        "per_threat": per_threat,
    }


# --------------------------------------------------------------------------- #
# 검증 / 정규화 / 점수
# --------------------------------------------------------------------------- #
def validate_run(m: dict) -> Tuple[bool, List[str]]:
    warnings: List[str] = []
    valid = True
    if m["distance_m"] < MIN_VALID_DISTANCE:
        valid = False
        warnings.append(f"무효 런: 이동거리 {m['distance_m']}m (< {MIN_VALID_DISTANCE}m) — 멈춤/미완주로 추천 제외")
    if not m["reached"] and valid:
        warnings.append("미도착(reached=false) — 유효 데이터지만 완주 실패")
    if m["collisions"] > COLLISION_WARN:
        warnings.append(f"충돌 과다: {m['collisions']}회 — 주행 품질 낮음")
    if not m["yolo_counts"] and m["distance_m"] >= MIN_VALID_DISTANCE:
        warnings.append("비전(YOLO) 미수집 — W1 위협발견 과소평가 가능")
    if m["trajectory"] is None:
        warnings.append("궤적 미수집 — W2 노출시간/발각횟수 N/A (재주행 시 Part B 로깅으로 채워짐)")
    return valid, warnings


def _clamp01(v: float) -> float:
    return max(0.0, min(1.0, v))


def normalize(metrics_by_route: Dict[str, dict], exposures: Dict[str, Optional[dict]],
              mode: str) -> Dict[str, dict]:
    """루트별 위험 항목 raw값 추출 후 0~1 정규화. None 항목은 available=False."""
    raw: Dict[str, Dict[str, Optional[float]]] = {}
    for rid, m in metrics_by_route.items():
        exp = exposures.get(rid)
        raw[rid] = {
            "threat": m["threat_proxy"],
            "exposure": (exp["total_fov_dwell_s"] if exp else None),
            "detour": (m["detour_ratio"] if m["detour_ratio"] is not None else None),
            "terrain": m["terrain_sigma"],
            "time": m["sim_time_s"],
        }

    out: Dict[str, dict] = {}
    keys = ["threat", "exposure", "detour", "terrain", "time"]
    for rid in metrics_by_route:
        out[rid] = {}
    for key in keys:
        vals = {rid: raw[rid][key] for rid in raw}
        avail = {rid: v for rid, v in vals.items() if v is not None}
        for rid in raw:
            v = vals[rid]
            if v is None:
                out[rid][key] = {"raw": None, "norm": None, "available": False}
                continue
            out[rid][key] = {"raw": round(v, 3), "norm": _norm_value(key, v, avail, mode), "available": True}
    return out


def _norm_value(key: str, v: float, avail_vals: Dict[str, float], mode: str) -> float:
    if mode == "minmax":
        lo, hi = min(avail_vals.values()), max(avail_vals.values())
        if hi - lo < 1e-9:
            return 0.5
        return round(_clamp01((v - lo) / (hi - lo)), 3)
    # reference
    if key == "detour":
        ref = REFS["detour"]
        return round(_clamp01((v - 1.0) / (ref - 1.0)) if ref > 1.0 else 0.5, 3)
    ref = REFS.get(key, 1.0)
    return round(_clamp01(v / ref) if ref > 0 else 0.5, 3)


def risk_score(norm_route: Dict[str, dict], weights: Dict[str, float]) -> Tuple[float, List[dict]]:
    """가중합 위험도 총점 + 항목별 분해."""
    keymap = [("threat", "W1", "적/초소 발견"), ("exposure", "W2", "시야 노출시간"),
              ("detour", "W3", "우회/이탈"), ("terrain", "W4", "지형 굴곡도"),
              ("time", "W5", "소요시간")]
    total = 0.0
    breakdown: List[dict] = []
    for key, wkey, label in keymap:
        cell = norm_route[key]
        w = weights[wkey]
        if cell["available"] and cell["norm"] is not None:
            contrib = round(w * cell["norm"], 4)
            total += contrib
        else:
            contrib = None
        breakdown.append({"key": wkey, "label": label, "weight": w,
                          "raw": cell["raw"], "norm": cell["norm"],
                          "contrib": contrib, "available": cell["available"]})
    return round(total, 4), breakdown


def recommend(scored: Dict[str, dict], tie_eps: float = 0.05) -> dict:
    valid = {rid: s for rid, s in scored.items() if s["valid"]}
    if not valid:
        return {"winner": None, "confidence": "none",
                "reason": "유효 런 없음 — 모든 루트가 무효(멈춤/미완주). 깨끗한 A/B 재주행 필요."}
    ranked = sorted(valid.items(), key=lambda kv: kv[1]["risk_total"])
    winner = ranked[0][0]
    if len(valid) == 1:
        return {"winner": winner, "confidence": "low",
                "reason": f"유효 루트가 route_{winner} 하나뿐 — 비교군 부재(상대 루트 무효)로 단독 평가."}
    second = ranked[1]
    gap = second[1]["risk_total"] - ranked[0][1]["risk_total"]
    if gap < tie_eps:
        return {"winner": winner, "confidence": "low",
                "reason": f"위험도 차 {gap:.3f} < {tie_eps} — 사실상 동률. 보조지표(충돌/지형)로 신중 판단 필요."}
    return {"winner": winner, "confidence": "medium",
            "reason": f"route_{winner} 위험도 최저({ranked[0][1]['risk_total']:.3f} vs {second[1]['risk_total']:.3f}, 차 {gap:.3f})."}


# --------------------------------------------------------------------------- #
# 센서 정확도(GT vs 탐지)
# --------------------------------------------------------------------------- #
def compute_perception_accuracy(m: dict, gt_objects: List[Dict[str, Any]]) -> dict:
    """LiDAR 탐지 centroid ↔ GT 정적객체 최근접 매칭 + YOLO 카운트 vs GT."""
    detected = m["obstacles_detected"]
    det_pts: List[Tuple[float, float]] = []
    for d in detected:
        try:
            det_pts.append((float(d["x"]), float(d["z"])))
        except (KeyError, TypeError, ValueError):
            continue

    gt_pts = [(o["x"], o["z"]) for o in gt_objects]
    matched_det = 0
    sq_errs: List[float] = []
    covered_gt = set()
    for (dx, dz) in det_pts:
        best_i, best_d = -1, float("inf")
        for gi, (gx, gz) in enumerate(gt_pts):
            dd = math.hypot(dx - gx, dz - gz)
            if dd < best_d:
                best_d, best_i = dd, gi
        if best_i >= 0 and best_d <= PERCEPTION_MATCH_TOL:
            matched_det += 1
            covered_gt.add(best_i)
            sq_errs.append(best_d * best_d)

    n_det = len(det_pts)
    n_gt = len(gt_pts)
    precision = round(matched_det / n_det, 3) if n_det else None
    recall = round(len(covered_gt) / n_gt, 3) if n_gt else None
    rmse = round(math.sqrt(sum(sq_errs) / len(sq_errs)), 3) if sq_errs else None

    # YOLO 카운트 vs GT(정적) — 카운트는 탐지 이벤트라 1:1 아님(참고용)
    from collections import Counter
    gt_prefix = Counter()
    for o in gt_objects:
        name = o["prefabName"]
        for yolo_cls, prefix in YOLO_TO_GT_PREFAB.items():
            if name.startswith(prefix):
                gt_prefix[yolo_cls] += 1
    yolo_vs_gt = []
    for cls in YOLO_TO_GT_PREFAB:
        yolo_vs_gt.append({"class": cls,
                           "yolo_detections": int(m["yolo_counts"].get(cls, 0)),
                           "gt_static": int(gt_prefix.get(cls, 0))})

    return {
        "lidar": {"detections": n_det, "gt_static_objects": n_gt,
                  "matched": matched_det, "covered_gt": len(covered_gt),
                  "precision": precision, "recall": recall, "position_rmse_m": rmse},
        "yolo_vs_gt": yolo_vs_gt,
    }


# --------------------------------------------------------------------------- #
# 렌더링
# --------------------------------------------------------------------------- #
def _setup_korean_font() -> None:
    """matplotlib 한글 라벨이 □로 깨지지 않게 Noto Sans CJK 등록(없으면 폴백)."""
    from matplotlib import font_manager
    import matplotlib.pyplot as plt
    for cjk in ("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
                "/usr/share/fonts/truetype/nanum/NanumGothic.ttf"):
        if os.path.exists(cjk):
            try:
                font_manager.fontManager.addfont(cjk)
                plt.rcParams["font.family"] = font_manager.FontProperties(fname=cjk).get_name()
                break
            except Exception:
                pass
    plt.rcParams["axes.unicode_minus"] = False


def render_exposure_figure(rid: str,
                           trajectory: Optional[List[Tuple[float, float, float]]],
                           threats: List[Dict[str, Any]],
                           gt_objects: List[Dict[str, Any]],
                           gt_bboxes: List[Dict[str, float]],
                           exposure: Optional[dict],
                           out_dir: str) -> Optional[str]:
    """실제 주행 궤적 + 초소(House002) 시야(FOV)/반경 + 실제 발각 지점을 PNG로 그린다.

    발각 판정은 보고서 수치와 동일하게 tg.is_threat_active(반경+FOV±30°+LoS)를 그대로 쓴다.
    matplotlib이 없거나 궤적 미수집이면 None을 반환(보고서는 그림 없이 계속).
    """
    if not trajectory:
        return None
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Wedge, Circle
    except Exception as e:  # pragma: no cover - 환경 의존
        print(f"[경고] matplotlib 미설치 — 노출 지도 생략: {e}", file=sys.stderr)
        return None
    _setup_korean_font()

    xs = [float(p[1]) for p in trajectory]
    zs = [float(p[2]) for p in trajectory]

    fig, ax = plt.subplots(figsize=(10, 10), dpi=130)

    # 1) GT 장애물(맥락) — 위협 아닌 정적객체만 옅은 점으로
    cls_color = {"Tree": "#7fae7f", "Rock": "#a9742f", "Car": "#ff8c1a",
                 "Tent": "#c2a36b", "Human": "#cfcfcf"}
    for o in gt_objects:
        if o.get("is_threat"):
            continue
        name = str(o.get("prefabName", ""))
        cls = next((k for k in cls_color if name.startswith(k)), None)
        if cls is None:
            continue
        ax.scatter([o["x"]], [o["z"]], s=12, color=cls_color[cls], alpha=0.45, zorder=1)

    # 2) 위협 시야: House002=부채꼴(25m·±30°), 그 외=반경 원
    for th in threats:
        tx, tz = float(th["x"]), float(th["z"])
        if str(th.get("type")) == "House002" or str(th.get("prefabName", "")).startswith("House002"):
            center_mpl = 90.0 - float(th.get("yaw", 0.0))   # atan2(dx,dz)→matplotlib(+x,CCW): 90-yaw
            ax.add_patch(Wedge((tx, tz), tg.HOUSE_RADIUS_M,
                               center_mpl - tg.HOUSE_FOV_HALF_DEG, center_mpl + tg.HOUSE_FOV_HALF_DEG,
                               facecolor="#d62728", alpha=0.13, edgecolor="#d62728", lw=0.8, zorder=2))
        else:
            ax.add_patch(Circle((tx, tz), tg.threat_radius(th),
                                facecolor="#d62728", alpha=0.10, edgecolor="#d62728", lw=0.8, zorder=2))
        ax.scatter([tx], [tz], marker="s", s=70, facecolors="none",
                   edgecolors="#d62728", linewidths=1.6, zorder=5)
        ax.annotate(str(th.get("prefabName", "")), (tx, tz), fontsize=6, color="#a01b1c",
                    xytext=(3, 3), textcoords="offset points", zorder=6)

    # 3) 실제 주행 궤적 + 진행 방향 화살표(연속점에서 유도 — 로깅 heading 의존 X)
    ax.plot(xs, zs, "-", color="#1f6fd6", lw=1.8, alpha=0.9, zorder=4, label="실제 주행 궤적")
    step = max(1, len(trajectory) // 12)
    for i in range(step, len(trajectory), step):
        ax.annotate("", xy=(xs[i], zs[i]), xytext=(xs[i - 1], zs[i - 1]),
                    arrowprops=dict(arrowstyle="->", color="#1f6fd6", alpha=0.6), zorder=4)

    # 4) 발각 지점(빨강) + 발각 초소로 연결선 — 보고서 노출시간과 동일 판정
    ex, ez = [], []
    for p in trajectory:
        x, z = float(p[1]), float(p[2])
        for th in threats:
            if tg.is_threat_active((x, z), th, gt_bboxes):
                ex.append(x)
                ez.append(z)
                ax.plot([x, float(th["x"])], [z, float(th["z"])],
                        color="#d62728", lw=0.5, alpha=0.30, zorder=3)
                break
    if ex:
        ax.scatter(ex, ez, s=26, color="#d62728", edgecolors="black", linewidths=0.4,
                   zorder=7, label=f"발각 샘플({len(ex)})")

    # 5) 출발/목적
    ax.scatter([xs[0]], [zs[0]], marker="o", s=130, color="#1a9850",
               edgecolors="black", zorder=8, label="출발")
    ax.scatter([GOAL_XY[0]], [GOAL_XY[1]], marker="^", s=150, color="black",
               edgecolors="white", zorder=8, label="목적지")

    det = exposure["detection_count"] if exposure else 0
    dwell = exposure["total_fov_dwell_s"] if exposure else 0.0
    ax.set_title(f"route_{rid} 실주행·위협 노출 — 발각 {det}회 · 노출 {dwell:.2f}s", fontsize=11)
    ax.set_xlim(0, 300)
    ax.set_ylim(0, 300)
    ax.set_aspect("equal")
    ax.set_xlabel("map x (동→) [m]")
    ax.set_ylabel("map z (북↑, 위쪽이 목적지) [m]")
    ax.grid(True, alpha=0.2)
    ax.legend(loc="upper left", fontsize=8, framealpha=0.9)
    out = os.path.join(out_dir, f"exposure_{rid}.png")
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def render_route_overview(map_path: str, routes_path: str, out_dir: str) -> Optional[str]:
    """A/B '계획 경로'를 새 맵 위에 1장으로 그린다(scripts/visualize_routes.py 재사용).

    실주행 노출 지도(5장)와 달리 '설계된' 전역경로를 맥락으로 보여준다(런타임과 동일하게
    웨이포인트+목적지 A*). 모듈/맵 로드 실패 시 None을 반환(보고서는 그림 없이 계속).
    """
    try:
        import yaml as _yaml
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))          # scripts/
        sys.path.insert(0, os.path.join(PROJECT_ROOT, "src", "path_planning"))   # team_path_planning
        from path_planning.team_path_planning import load_static_obstacles_from_map
        import visualize_routes as vr
    except Exception as e:  # pragma: no cover - 환경 의존
        print(f"[경고] 계획 경로 지도 생략(모듈 로드 실패): {e}", file=sys.stderr)
        return None
    try:
        planning_obs = load_static_obstacles_from_map(map_path)
        raw_objs = vr.load_raw_objects(map_path)
        with open(routes_path, encoding="utf-8") as f:
            rd = _yaml.safe_load(f)["finalmap"]
        start = tuple(rd["start"])
        goal = tuple(rd["destination"])
        routes_paths: Dict[str, Any] = {}
        wps: Dict[str, Any] = {}
        for rid, side, color in (("A", "west", "#1f77b4"), ("B", "east", "#2ca02c")):
            wp = [tuple(p) for p in rd["routes"][rid]]
            wps[rid] = wp
            routes_paths[rid] = (vr.plan_route(start, wp, goal, side, planning_obs, 5.0, 0.4), color)
        out = os.path.join(out_dir, "route_overlay.png")
        vr.render_overlay(out, planning_obs, raw_objs, routes_paths, start, goal, wps)
        return out
    except Exception as e:  # pragma: no cover
        print(f"[경고] 계획 경로 지도 생성 실패: {e}", file=sys.stderr)
        return None


def _fmt(v: Any, na: str = "—") -> str:
    return na if v is None else (f"{v:g}" if isinstance(v, float) else str(v))


def _pick_better(a: Any, b: Any, lower_better: bool = True) -> Tuple[str, str]:
    """두 값 중 더 나은 쪽에 ✓. None은 비교 제외."""
    if a is None or b is None or a == b:
        return "", ""
    a_better = (a < b) if lower_better else (a > b)
    return ("✓", "") if a_better else ("", "✓")


def render_markdown(routes: List[str], metrics: Dict[str, dict], exposures: Dict[str, Optional[dict]],
                    scored: Dict[str, dict], perception: Dict[str, dict],
                    rec: dict, weights: Dict[str, float], norm_mode: str,
                    figures: Optional[Dict[str, Optional[str]]] = None,
                    overview_fig: Optional[str] = None) -> str:
    L: List[str] = []
    rk = " / ".join(f"route_{r}" for r in routes)
    L.append(f"# 정찰 보고서 — {rk}")
    L.append("")
    L.append("> 정밀 정찰 및 경로 위험도·은밀성 평가 (Scenario 1). 값은 **↓ 낮을수록 안전(은밀)**.")
    L.append("")

    # 1. 임무 개요
    L.append("## 1. 임무 개요")
    L.append("")
    mp = next((metrics[r]["map"] for r in routes if metrics[r]["map"]), "finalmap")
    L.append(f"- 맵: `{mp}` · 출발 {START_XY} → 목적지 {GOAL_XY}")
    L.append(f"- 위험도 가중치: " + ", ".join(f"{k}={v}" for k, v in weights.items()) + f" · 정규화: `{norm_mode}`")
    L.append(f"- 입력 루트: {', '.join('route_' + r for r in routes)}")
    L.append("")
    if overview_fig:
        L.append(f"![A/B 계획 경로 (새 맵)]({os.path.basename(overview_fig)})")
        L.append("")
        L.append("> A(서·파랑)/B(동·초록) **계획 경로**를 맵 위에 표시 (▲=목적지, ★=적전차 리스폰). "
                 "실제 주행 궤적·발각은 5장 참고.")
        L.append("")

    # 2. 주행 요약
    L.append("## 2. 주행 요약")
    L.append("")
    L.append("| 항목 | " + " | ".join(f"route_{r}" for r in routes) + " |")
    L.append("|---|" + "---|" * len(routes))
    def row(name, fn):
        return f"| {name} | " + " | ".join(fn(metrics[r]) for r in routes) + " |"
    L.append(row("도착(reached)", lambda m: "✅" if m["reached"] else "❌"))
    L.append(row("유효 런", lambda m: ("✅" if scored[m['route']]['valid'] else "⚠️ 무효")))
    L.append(row("이동거리 (m)", lambda m: _fmt(m["distance_m"])))
    L.append(row("소요시간 (s)", lambda m: _fmt(m["sim_time_s"])))
    L.append(row("충돌 횟수", lambda m: _fmt(m["collisions"])))
    L.append(row("장애물 밀도 (/100m)", lambda m: _fmt(m["obstacle_density"])))
    L.append("")

    # 3. 위협·객체 발견
    L.append("## 3. 위협·객체 발견 (YOLO)")
    L.append("")
    all_cls = sorted({c for r in routes for c in metrics[r]["yolo_counts"]})
    if all_cls:
        L.append("| 클래스 | " + " | ".join(f"route_{r}" for r in routes) + " | 위협가중 |")
        L.append("|---|" + "---|" * len(routes) + "---|")
        for c in all_cls:
            w = THREAT_CLASS_WEIGHTS.get(c)
            wtxt = f"×{w}" if w else "(비위협)"
            L.append(f"| {c} | " + " | ".join(str(metrics[r]["yolo_counts"].get(c, 0)) for r in routes) + f" | {wtxt} |")
    else:
        L.append("_YOLO 탐지 데이터 없음._")
    L.append("")

    # 4. 정량 비교표
    L.append("## 4. 정량 비교표 (↓ 낮을수록 좋음)")
    L.append("")
    if len(routes) == 2:
        a, b = routes
        L.append(f"| 지표 | route_{a} | route_{b} |")
        L.append("|---|---|---|")
        def cmp_row(name, va, vb, lower=True):
            ma, mb = _pick_better(va, vb, lower)
            return f"| {name} | {_fmt(va)} {ma} | {_fmt(vb)} {mb} |"
        L.append(cmp_row("위험도 총점", scored[a]["risk_total"], scored[b]["risk_total"]))
        L.append(cmp_row("위협 발견(W1 가중)", metrics[a]["threat_proxy"], metrics[b]["threat_proxy"]))
        ea, eb = exposures.get(a), exposures.get(b)
        L.append(cmp_row("시야 노출시간 s(W2)", ea["total_fov_dwell_s"] if ea else None, eb["total_fov_dwell_s"] if eb else None))
        L.append(cmp_row("발각 횟수", ea["detection_count"] if ea else None, eb["detection_count"] if eb else None))
        L.append(cmp_row("우회비(W3)", metrics[a]["detour_ratio"], metrics[b]["detour_ratio"]))
        L.append(cmp_row("지형 굴곡 σ(W4)", metrics[a]["terrain_sigma"], metrics[b]["terrain_sigma"]))
        L.append(cmp_row("소요시간 s(W5)", metrics[a]["sim_time_s"], metrics[b]["sim_time_s"]))
        L.append(cmp_row("충돌(참고)", metrics[a]["collisions"], metrics[b]["collisions"]))
    else:
        L.append("_단일 루트 — 비교표 생략._")
    L.append("")

    # 5. 위험도·은밀성 점수 분해
    L.append("## 5. 위험도·은밀성 점수 분해")
    L.append("")
    if figures and any(v for v in figures.values()):
        L.append("> 각 루트 지도: **파란 선** = 실제 주행 궤적, **빨강 부채꼴** = 초소(House002) 시야"
                 "(반경 25m·정면 ±30°·시선차단 반영), **빨강 점** = 실제로 시야에 들어 발각된 샘플.")
        L.append("")
    for r in routes:
        L.append(f"### route_{r} — 위험도 총점 **{scored[r]['risk_total']:.3f}**" +
                 ("" if scored[r]["valid"] else " ⚠️(무효 런)"))
        L.append("")
        L.append("| 항목 | 가중 | 원시값 | 정규화 | 기여 |")
        L.append("|---|---|---|---|---|")
        for b in scored[r]["breakdown"]:
            raw = _fmt(b["raw"]) if b["available"] else "**N/A**"
            norm = _fmt(b["norm"]) if b["available"] else "—"
            contrib = _fmt(b["contrib"]) if b["available"] else "0"
            L.append(f"| {b['key']} {b['label']} | {b['weight']} | {raw} | {norm} | {contrib} |")
        L.append("")
        exp = exposures.get(r)
        if exp and exp["per_threat"]:
            L.append("발각된 위협(초소/적전차) — 노출시간 상위:")
            L.append("")
            L.append("| 위협 | 발각횟수 | 노출시간 s | 최대연속 s | 최소거리 m |")
            L.append("|---|---|---|---|---|")
            for e in exp["per_threat"][:6]:
                L.append(f"| {e['threat']} | {e['detections']} | {e['fov_dwell_s']} | {e['max_continuous_s']} | {_fmt(e['min_dist_m'])} |")
            L.append("")

        fig = (figures or {}).get(r)
        if fig:
            L.append(f"![route_{r} 실주행·위협 노출 지도]({os.path.basename(fig)})")
            L.append("")

    # 6. 센서 인지 정확도
    L.append("## 6. 센서 인지 정확도 (GT vs 탐지)")
    L.append("")
    L.append("> GT = `finalmap.map` 정적 객체. 런타임 스폰 객체는 정적 GT에 없을 수 있어 **정적 GT 한정** 지표다.")
    L.append("")
    for r in routes:
        p = perception[r]["lidar"]
        L.append(f"### route_{r} — LiDAR")
        L.append(f"- 탐지 {p['detections']}건 / GT 정적객체 {p['gt_static_objects']}개 · "
                 f"매칭 {p['matched']} · GT 커버 {p['covered_gt']}")
        L.append(f"- precision {_fmt(p['precision'])} · recall {_fmt(p['recall'])} · "
                 f"위치 RMSE {_fmt(p['position_rmse_m'])} m (허용오차 {PERCEPTION_MATCH_TOL}m)")
        yv = perception[r]["yolo_vs_gt"]
        L.append("- YOLO 탐지수 vs GT(정적): " +
                 ", ".join(f"{x['class']} {x['yolo_detections']}/{x['gt_static']}" for x in yv))
        L.append("")

    # 7. 최종 추천
    L.append("## 7. 최종 추천")
    L.append("")
    if rec["winner"]:
        L.append(f"### 🏆 권장 루트: **route_{rec['winner']}**  (신뢰도: {rec['confidence']})")
    else:
        L.append("### ⚠️ 추천 불가")
    L.append("")
    L.append(f"- 근거: {rec['reason']}")
    L.append("")

    # 8. 데이터 품질·미구현 경고
    L.append("## 8. 데이터 품질 · 미구현 경고")
    L.append("")
    any_warn = False
    for r in routes:
        ws = scored[r]["warnings"]
        if ws:
            any_warn = True
            L.append(f"- **route_{r}**")
            for w in ws:
                L.append(f"  - {w}")
    if not any_warn:
        L.append("- (없음)")
    L.append("")
    L.append("- W3 우회는 재계획 카운트 미로깅으로 `distance/직선거리` **근사**다.")
    L.append("- 노출/발각은 GT 위협 + 전차 궤적 기반 사후계산이며, House002는 FOV±30°+LoS, Tank001은 반경+LoS 모델을 따른다.")
    L.append("")
    return "\n".join(L)


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="정찰 보고서 생성기 (recon_reports/*.json → recon_report.md)")
    p.add_argument("--input", default=DEFAULT_REPORT_DIR,
                   help="comparison.json 또는 route_*.json 이 있는 디렉터리/파일 (기본 recon_reports/)")
    p.add_argument("--route-a", default=None, help="route_A.json 직접 지정(폴백)")
    p.add_argument("--route-b", default=None, help="route_B.json 직접 지정(폴백)")
    p.add_argument("-o", "--output", default=None, help="출력 md 경로 (기본 <input>/recon_report.md)")
    p.add_argument("--map", default=DEFAULT_MAP, help="GT 맵 파일 (기본 finalmap.map)")
    p.add_argument("--routes", default=DEFAULT_ROUTES,
                   help="계획 경로 개요 그림용 routes.yaml (기본 path_planning/config/routes.yaml)")
    p.add_argument("--norm", choices=["reference", "minmax"], default="reference", help="정규화 방식")
    for k in ("w1", "w2", "w3", "w4", "w5"):
        p.add_argument(f"--{k}", type=float, default=None, help=f"가중치 {k.upper()} 오버라이드")
    p.add_argument("--stdout", action="store_true", help="파일 대신 표준출력")
    p.add_argument("--no-figures", action="store_true",
                   help="노출 지도 PNG(exposure_*.png) 생성/임베드 생략")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)

    weights = dict(DEFAULT_WEIGHTS)
    for i, k in enumerate(("w1", "w2", "w3", "w4", "w5"), start=1):
        v = getattr(args, k)
        if v is not None:
            weights[f"W{i}"] = v

    routes_data = load_inputs(args.input, args.route_a, args.route_b)
    if not routes_data:
        print(f"[오류] 입력을 찾지 못함: {args.input} (comparison.json / route_*.json 없음)", file=sys.stderr)
        return EXIT_NO_INPUT

    # GT 맵 로드(없어도 진행 — 노출/센서정확도만 비활성)
    try:
        threats, gt_bboxes, gt_objects = tg.load_map(args.map)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"[경고] GT 맵 로드 실패({e}) — 노출/센서정확도 생략", file=sys.stderr)
        threats, gt_bboxes, gt_objects = [], [], []

    routes = sorted(routes_data.keys())
    metrics: Dict[str, dict] = {}
    exposures: Dict[str, Optional[dict]] = {}
    perception: Dict[str, dict] = {}
    scored: Dict[str, dict] = {}

    for rid in routes:
        m = extract_metrics(routes_data[rid])
        metrics[rid] = m
        exposures[rid] = compute_exposure(m["trajectory"], threats, gt_bboxes)
        perception[rid] = compute_perception_accuracy(m, gt_objects)

    norm = normalize(metrics, exposures, args.norm)
    for rid in routes:
        valid, warns = validate_run(metrics[rid])
        total, breakdown = risk_score(norm[rid], weights)
        scored[rid] = {"risk_total": total, "breakdown": breakdown,
                       "valid": valid, "warnings": warns}

    rec = recommend(scored)

    # 노출 지도(PNG)는 md와 같은 폴더에 생성해 basename으로 임베드한다. stdout/--no-figures면 생략.
    figures: Dict[str, Optional[str]] = {}
    overview_fig: Optional[str] = None
    out = None
    if not args.stdout:
        out = args.output or os.path.join(
            args.input if os.path.isdir(args.input) else os.path.dirname(args.input) or ".",
            "recon_report.md")
        if not args.no_figures:
            out_dir = os.path.dirname(out) or "."
            overview_fig = render_route_overview(args.map, args.routes, out_dir)
            for rid in routes:
                figures[rid] = render_exposure_figure(
                    rid, metrics[rid]["trajectory"], threats, gt_objects, gt_bboxes,
                    exposures[rid], out_dir)

    md = render_markdown(routes, metrics, exposures, scored, perception, rec, weights,
                         args.norm, figures, overview_fig)

    if args.stdout:
        print(md)
    else:
        with open(out, "w", encoding="utf-8") as f:
            f.write(md)
        print(f"[완료] 정찰 보고서 작성: {out}")
        made = ([f"계획경로 1"] if overview_fig else []) + \
               ([f"노출 지도 {sum(1 for v in figures.values() if v)}"] if any(figures.values()) else [])
        if made:
            print("  그림: " + " · ".join(made) + " (recon_reports/*.png)")
        if rec["winner"]:
            print(f"  추천: route_{rec['winner']} (신뢰도 {rec['confidence']})")

    # 종료코드
    if not any(s["valid"] for s in scored.values()):
        return EXIT_NO_VALID_RUN
    if len(routes) < 2:
        return EXIT_SINGLE_ROUTE
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
