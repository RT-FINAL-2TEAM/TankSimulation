# -*- coding: utf-8 -*-
"""기하 위험도 — 새 적전차 출현 시 돌파/복귀 결정의 빠른·안전망 점수(0~1).

SCENARIO2_DESIGN §6: 거리 + 시선차단(LoS)으로 위험도 score 산정. Tank는 heading이 없어
반경+LoS만 본다(FOV 콘 없음, `is_threat_active`의 Tank001 규칙과 동일).

LoS 기하는 `scripts/recon_eval/threat_geometry.py`(검증된 위협 기하의 ROS-free 미러)를 재활용한다 —
세 번째 사본을 만들지 않기 위함(단일 출처 지향). scripts/recon_eval 는 ROS 패키지가 아니라
import path에 없으므로, 설치된 노드에서도 찾도록 경로를 자동 탐색한다(env → 심링크 실경로 walk-up).
탐색 실패 시 LoS 항을 건너뛰고 거리만으로 평가(graceful degrade) — 호출측이 1회 경고.

위험도(다중 위협은 max — 가장 위험한 하나가 임계를 넘으면 복귀):
  dist > radius            → 0.0  (위협 반경 밖)
  그 외                    → clamp01(w_dist*(1 - dist/radius) + w_los*(1.0 if LoS else 0.0))
"""

from __future__ import annotations

import math
import os
import sys
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple


def _find_recon_eval_dir() -> Optional[str]:
    """threat_geometry.py 가 있는 scripts/recon_eval 디렉터리를 찾는다."""
    candidates: List[str] = []
    env_root = os.environ.get("TANK_PROJECT_ROOT")
    if env_root:
        candidates.append(os.path.join(env_root, "scripts", "recon_eval"))
    # 이 파일의 실경로(심링크 해제)에서 위로 올라가며 탐색.
    # symlink-install이면 install/.../risk.py → src/mission/mission/risk.py 로 resolve된다.
    here = os.path.realpath(__file__)
    d = here
    for _ in range(8):
        d = os.path.dirname(d)
        if not d or d == os.path.dirname(d):
            break
        candidates.append(os.path.join(d, "scripts", "recon_eval"))
    for c in candidates:
        if os.path.isfile(os.path.join(c, "threat_geometry.py")):
            return c
    return None


@lru_cache(maxsize=1)
def _load_geo():
    """threat_geometry 모듈을 import한다. 실패하면 None(거리만 사용)."""
    path = _find_recon_eval_dir()
    if path and path not in sys.path:
        sys.path.insert(0, path)
    try:
        import threat_geometry as tg  # type: ignore
        return tg
    except Exception:
        return None


def los_available() -> bool:
    """LoS 기하(threat_geometry)를 쓸 수 있는지."""
    return _load_geo() is not None


def bboxes_from_map(scenario2_map: Dict[str, Any]) -> List[Dict[str, float]]:
    """scenario2_map.map(또는 finalmap)의 obstacles[] → LoS 차폐용 축정렬 bbox 리스트.

    threat_geometry.obstacle_to_bbox 재활용. geo 미사용이면 빈 리스트(LoS 항 0).
    """
    tg = _load_geo()
    if tg is None:
        return []
    bboxes: List[Dict[str, float]] = []
    for obs in scenario2_map.get("obstacles", []) or []:
        if not isinstance(obs, dict):
            continue
        bbox = tg.obstacle_to_bbox(obs)
        if bbox is not None:
            bboxes.append(bbox)
    return bboxes


def check_los(player_xy: Tuple[float, float], target_xy: Tuple[float, float],
              gt_bboxes: List[Dict[str, float]]) -> bool:
    """전차(player)→표적 직선이 장애물에 안 가리면 True. geo 없으면 보수적으로 True(노출 가정)."""
    tg = _load_geo()
    if tg is None:
        return True
    # threat_geometry.check_los 는 (관측자_x, 관측자_z, 표적_x, 표적_z, bboxes).
    # map 평면에서 모듈의 z축 = 진행축(map.y)이므로 (x, y)를 (x, z)로 그대로 전달한다.
    return bool(tg.check_los(player_xy[0], player_xy[1], target_xy[0], target_xy[1], gt_bboxes))


def _clamp01(v: float) -> float:
    return max(0.0, min(1.0, v))


def geometric_risk_score(player_xy: Tuple[float, float], tank_xy: Tuple[float, float],
                         gt_bboxes: List[Dict[str, float]],
                         radius_m: float = 20.0,
                         w_dist: float = 0.6, w_los: float = 0.4) -> float:
    """단일 새 적전차에 대한 0~1 위험도. 반경 밖이면 0."""
    dist = math.hypot(player_xy[0] - tank_xy[0], player_xy[1] - tank_xy[1])
    if dist > radius_m:
        return 0.0
    dist_term = 1.0 - (dist / radius_m) if radius_m > 0 else 0.0
    los_term = 1.0 if check_los(player_xy, tank_xy, gt_bboxes) else 0.0
    return _clamp01(w_dist * dist_term + w_los * los_term)


def worst_risk(player_xy: Tuple[float, float],
               tanks_xy: List[Tuple[float, float]],
               gt_bboxes: List[Dict[str, float]],
               radius_m: float = 20.0,
               w_dist: float = 0.6, w_los: float = 0.4
               ) -> Tuple[float, Optional[Tuple[float, float]]]:
    """여러 새 적전차 중 최대 위험도와 그 표적 좌표를 돌려준다(없으면 (0.0, None))."""
    best_score = 0.0
    best_xy: Optional[Tuple[float, float]] = None
    for t in tanks_xy:
        s = geometric_risk_score(player_xy, t, gt_bboxes, radius_m, w_dist, w_los)
        if s > best_score:
            best_score = s
            best_xy = t
    return best_score, best_xy
