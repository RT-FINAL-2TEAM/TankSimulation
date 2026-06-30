import json
import os
import textwrap
from datetime import datetime

import requests


class LLMReporter:
    def __init__(
        self,
        ollama_url: str = None,
        model_name: str = None,
        timeout_sec: int = None,
    ):
        # 모델/URL/timeout은 env 단일 출처(나중에 .env만 바꿔 GPU PC에서 큰 모델로 교체).
        self.ollama_url = ollama_url or os.environ.get(
            "TANK_OLLAMA_URL", "http://localhost:11434/api/generate"
        )
        self.model_name = model_name or os.environ.get(
            "TANK_LLM_MODEL", os.environ.get("TANK_OLLAMA_MODEL", "qwen3:0.6b")
        )
        try:
            self.timeout_sec = int(timeout_sec or os.environ.get("TANK_LLM_TIMEOUT_SEC", "1800"))
        except (TypeError, ValueError):
            self.timeout_sec = 1800

    def forced_route(self):
        raw = os.environ.get("TANK_FORCE_ROUTE", "A").strip().upper()
        if raw in {"", "0", "FALSE", "NO", "NONE", "OFF", "AUTO"}:
            return None
        if raw in {"A", "B"}:
            return raw
        return None

    def build_prompt(self, comparison_data: dict) -> str:
        forced_route = self.forced_route()
        force_instruction = ""
        if forced_route:
            force_instruction = textwrap.dedent(f"""
            FORCED MISSION POLICY:
            - selected_route MUST be "{forced_route}".
            - Analyze risk_level, key_risks, and used_evidence from the real input values.
            - Even if another route appears safer, keep selected_route as "{forced_route}".
            - Do not mention that the final route was forced or fixed by policy.
            """).strip()

        return textwrap.dedent(f"""
        /no_think

        너는 전차 정찰 임무의 전술 참모 AI다.

        아래 입력은 A 루트와 B 루트의 정찰 결과다.
        반드시 입력 JSON에 있는 값만 근거로 사용하라.
        입력에 없는 숫자나 항목을 만들지 마라.

        판단 기준 (위험 축을 1차로, 효율은 보조로):
        - reached=false이면 정찰 미완주로 매우 불리하다.
        - [노출 = 1차 위험] stealth_ratio(0~1)는 적에게 실제로 보이며 주행하는 경로 길이 비율이다. 클수록 위험(은밀성 낮음).
        - proximity_ratio(0~1)는 적 탐지반경 안을 지나는 길이 비율이다. 클수록 위험.
        - exposure_available=false이면 노출 미측정이니 노출은 판단에서 제외한다.
        - closest_enemy_distance_m는 가장 가까운 적과의 거리(m)다. null이면 위협 미탐지로 보고 제외, 값이 있으면 작을수록 위험하다.
        - [위협 맥락] enemy_count는 센서퓨전으로 확정된 distinct 적/초소 수다(많을수록 위험). enemy_by_class로 초소(house)/전차(tank)를 구분하라. yolo_counts_raw는 중복 누적 탐지 프레임이니 적 수로 쓰지 마라.
        - [험지] pitch_std_deg·roll_std_deg(또는 terrain_sigma_deg)가 클수록 지형 주행 안정성이 낮다(위험).
        - [효율 = 보조, 위험 아님] distance_m·sim_time_s·detour_ratio·obstacle_count·obstacle_density_per_100m는 '이동 부담'이다. risk_level은 노출·위협·험지로 정하고, 효율은 위험이 비슷할 때만 보조로 써라.
        - [신뢰도] gt_confidence가 낮거나(<0.5) gt_found가 gt_total보다 많이 적으면 정찰이 위협을 놓쳤을 수 있으니 confidence를 낮춰라.
        - 단일 항목이 아니라 전체 위험 맥락을 비교해 selected_route를 선택하라.

        출력 규칙:
        - 반드시 JSON 객체 하나만 출력한다.
        - JSON 밖에 설명, 마크다운, 코드블록을 쓰지 마라.
        - selected_route는 반드시 "A" 또는 "B" 중 하나다. selected_route는 **실제 임무에 사용할 권장 루트 = 종합 위험이 더 낮은(더 안전한) 쪽**이다. 더 위험한 루트를 고르지 마라.
        - risk_level 값은 "low", "medium", "high", "critical" 중 하나다.
        - confidence 값은 "low", "medium", "high" 중 하나다.
        - speed_policy 값은 "slow", "medium", "fast" 중 하나다.
        - key_risks에는 필드명만 쓰지 말고 실제 위험 내용을 한국어로 설명하라.
        - used_evidence에는 입력 JSON의 실제 값을 그대로 복사하라.

        출력 JSON 구조:
        {{
        "selected_route": "A",
        "risk_level": {{
            "A": "low",
            "B": "medium"
        }},
        "confidence": "medium",
        "summary": "한국어 한 문장",
        "decision_reason": "선택 이유를 한국어로 구체적으로 설명",
        "key_risks": {{
            "A": ["A 루트 위험 요인"],
            "B": ["B 루트 위험 요인"]
        }},
        "recommended_behavior": {{
            "speed_policy": "slow",
            "caution_points": ["주의점"],
            "tactical_note": "전술 메모"
        }},
        "used_evidence": {{
            "A": {{ "route_id": "A", "reason": "" }},
            "B": {{ "route_id": "B", "reason": "" }}
        }}
        }}
        (used_evidence의 숫자는 시스템이 입력값으로 자동 채우니 reason만 쓰면 된다.)

        예시 (형식·판단 흐름 참고용 — 반드시 아래 '입력 데이터'의 실제 값으로 새로 판단하라):
        입력 예: route_A stealth_ratio=0.10, enemy_count=1, terrain_sigma_deg=8 / route_B stealth_ratio=0.55, enemy_count=3, terrain_sigma_deg=6
        출력 예: {{"selected_route":"A","risk_level":{{"A":"low","B":"high"}},"confidence":"medium","summary":"A는 노출이 낮아 더 은밀하다.","decision_reason":"B는 stealth_ratio 0.55로 경로 절반이 적 시야에 노출되고 확정 적이 3으로 더 많다. A는 노출 0.10·적 1로 은밀성이 높다. 지형은 A가 약간 거칠지만 위험축에서 노출 차이가 결정적이라 A를 택했다.","key_risks":{{"A":["지형 굴곡 다소 높음"],"B":["경로 절반이 적 시야에 노출","확정 적 3"]}},"recommended_behavior":{{"speed_policy":"medium","caution_points":["A 후반 험지 구간 감속"],"tactical_note":"A로 진입하되 노출 구간 진입 전 정지·관측."}},"used_evidence":{{"A":{{"route_id":"A","reason":"노출 최소"}},"B":{{"route_id":"B","reason":"노출 과다"}}}}}}

        {force_instruction}

        입력 데이터:
        {json.dumps(comparison_data, ensure_ascii=False, indent=2)}
        """).strip()

    def is_valid_result(self, parsed: dict) -> bool:
        required_keys = {
            "selected_route",
            "risk_level",
            "confidence",
            "summary",
            "decision_reason",
            "key_risks",
            "recommended_behavior",
            "used_evidence",
        }

        if not isinstance(parsed, dict):
            return False

        if not required_keys.issubset(parsed.keys()):
            return False

        if parsed.get("selected_route") not in {"A", "B"}:
            return False

        risk_level = parsed.get("risk_level")
        if not isinstance(risk_level, dict):
            return False

        if risk_level.get("A") not in {"low", "medium", "high", "critical"}:
            return False

        if risk_level.get("B") not in {"low", "medium", "high", "critical"}:
            return False

        if parsed.get("confidence") not in {"low", "medium", "high"}:
            return False

        recommended = parsed.get("recommended_behavior")
        if not isinstance(recommended, dict):
            return False

        if recommended.get("speed_policy") not in {"slow", "medium", "fast"}:
            return False

        return True

    def fallback_result(self, raw_text: str) -> dict:
        return {
            "selected_route": None,
            "risk_level": {
                "A": "high",
                "B": "high",
            },
            "confidence": "low",
            "summary": "LLM 응답이 유효한 위험도 분석 JSON 형식을 만족하지 못했습니다.",
            "decision_reason": raw_text,
            "key_risks": {
                "A": [],
                "B": [],
            },
            "recommended_behavior": {
                "speed_policy": "slow",
                "caution_points": ["LLM 응답 검증 실패"],
                "tactical_note": "원본 raw_text와 입력 데이터를 확인해야 합니다.",
            },
            "used_evidence": {},
        }

    def validate_and_fix_result(self, parsed: dict, comparison_data: dict, apply_forced_route: bool = True) -> dict:
        allowed_routes = {"A", "B"}
        allowed_risk = {"low", "medium", "high", "critical"}
        allowed_confidence = {"low", "medium", "high"}
        allowed_speed = {"slow", "medium", "fast"}

        if not isinstance(parsed, dict):
            parsed = self.fallback_result("parsed result is not dict")

        if parsed.get("selected_route") not in allowed_routes:
            parsed["selected_route"] = None

        risk_level = parsed.get("risk_level")
        if not isinstance(risk_level, dict):
            risk_level = {}

        for route in ["A", "B"]:
            if risk_level.get(route) not in allowed_risk:
                # 위험도 분석 실패 시 안전 측면에서 high로 보정
                risk_level[route] = "high"

        parsed["risk_level"] = risk_level

        if parsed.get("confidence") not in allowed_confidence:
            parsed["confidence"] = "low"

        if not isinstance(parsed.get("summary"), str):
            parsed["summary"] = ""

        if not isinstance(parsed.get("decision_reason"), str):
            parsed["decision_reason"] = ""

        key_risks = parsed.get("key_risks")
        if not isinstance(key_risks, dict):
            key_risks = {}

        if not isinstance(key_risks.get("A"), list):
            key_risks["A"] = []

        if not isinstance(key_risks.get("B"), list):
            key_risks["B"] = []

        parsed["key_risks"] = key_risks

        recommended = parsed.get("recommended_behavior")
        if not isinstance(recommended, dict):
            recommended = {}

        if recommended.get("speed_policy") not in allowed_speed:
            recommended["speed_policy"] = "slow"

        if not isinstance(recommended.get("caution_points"), list):
            recommended["caution_points"] = []

        if not isinstance(recommended.get("tactical_note"), str):
            recommended["tactical_note"] = ""

        parsed["recommended_behavior"] = recommended

        route_a = comparison_data.get("route_A", {})
        route_b = comparison_data.get("route_B", {})

        old_evidence = parsed.get("used_evidence")
        if not isinstance(old_evidence, dict):
            old_evidence = {}

        old_a = old_evidence.get("A", {})
        if not isinstance(old_a, dict):
            old_a = {}

        old_b = old_evidence.get("B", {})
        if not isinstance(old_b, dict):
            old_b = {}

        # used_evidence의 숫자는 LLM 결과를 믿지 않고 입력 JSON(route_comparison)의 실제 값으로 덮어쓴다.
        def copy_evidence(route_data: dict, old_reason: str = "") -> dict:
            return {
                "route_id": route_data.get("route_id"),
                "reached": route_data.get("reached"),
                "enemy_count": route_data.get("enemy_count"),
                "enemy_by_class": route_data.get("enemy_by_class") if isinstance(route_data.get("enemy_by_class"), dict) else {},
                "closest_enemy_distance_m": route_data.get("closest_enemy_distance_m"),
                "stealth_ratio": route_data.get("stealth_ratio"),
                "proximity_ratio": route_data.get("proximity_ratio"),
                "exposure_available": route_data.get("exposure_available"),
                "distance_m": route_data.get("distance_m"),
                "sim_time_s": route_data.get("sim_time_s"),
                "detour_ratio": route_data.get("detour_ratio"),
                "collision_count": route_data.get("collision_count"),
                "obstacle_count": route_data.get("obstacle_count"),
                "obstacle_density_per_100m": route_data.get("obstacle_density_per_100m"),
                "pitch_std_deg": route_data.get("pitch_std_deg"),
                "roll_std_deg": route_data.get("roll_std_deg"),
                "terrain_sigma_deg": route_data.get("terrain_sigma_deg"),
                "gt_found": route_data.get("gt_found"),
                "gt_total": route_data.get("gt_total"),
                "gt_confidence": route_data.get("gt_confidence"),
                "yolo_counts_raw": route_data.get("yolo_counts_raw") if isinstance(route_data.get("yolo_counts_raw"), dict) else {},
                "asset_spotted_gt": route_data.get("asset_spotted_gt") if isinstance(route_data.get("asset_spotted_gt"), dict) else {},
                "reason": old_reason,
            }

        parsed["used_evidence"] = {
            "A": copy_evidence(route_a, old_a.get("reason", "")),
            "B": copy_evidence(route_b, old_b.get("reason", "")),
        }

        forced_route = self.forced_route() if apply_forced_route else None
        if forced_route:
            parsed["selected_route"] = forced_route
            parsed["confidence"] = "high"
            if not str(parsed.get("summary") or "").strip():
                parsed["summary"] = "A/B 루트 위험도 평가가 완료되었습니다."
            if not str(parsed.get("decision_reason") or "").strip():
                parsed["decision_reason"] = "노출 시간, 장애물 수, 차단 구간, 지형 안정성을 기준으로 평가했습니다."

            recommended = parsed.get("recommended_behavior")
            if not isinstance(recommended, dict):
                recommended = {}
            forced_level = str(parsed["risk_level"].get(forced_route) or "").lower()
            recommended["speed_policy"] = "slow" if forced_level in {"high", "critical"} else "medium"
            caution_points = recommended.get("caution_points")
            if not isinstance(caution_points, list):
                caution_points = []
            recommended["caution_points"] = caution_points
            if not str(recommended.get("tactical_note") or "").strip():
                recommended["tactical_note"] = "위험 지표가 높은 구간에서는 감속하고 장애물/노출 변화를 계속 확인합니다."
            parsed["recommended_behavior"] = recommended

        return parsed

    def call_ollama(self, prompt: str) -> dict:
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "format": "json",
            "options": {
                "temperature": 0,
                "num_predict": 768,
            },
        }

        response = requests.post(
            self.ollama_url,
            json=payload,
            timeout=self.timeout_sec,
        )
        response.raise_for_status()

        return response.json()

    def generate_route_decision(self, comparison_data: dict) -> dict:
        prompt = self.build_prompt(comparison_data)

        raw_text = ""
        parsed = None
        parsed_ok = False
        validated_ok = False
        retry_used = False

        try:
            ollama_result = self.call_ollama(prompt)
            raw_text = ollama_result.get("response", "")

            parsed = json.loads(raw_text)
            parsed_ok = True
            validated_ok = self.is_valid_result(parsed)

            # JSON 문법은 맞지만 {}처럼 필수 필드가 없는 경우 한 번 재시도
            if not validated_ok:
                retry_used = True

                retry_prompt = prompt + textwrap.dedent("""
                
                이전 응답은 필수 필드가 누락되어 실패했다.
                절대로 빈 JSON 객체 {}를 출력하지 마라.
                selected_route, risk_level, confidence, summary, decision_reason,
                key_risks, recommended_behavior, used_evidence를 모두 포함한
                완전한 JSON 객체를 다시 출력하라.
                """).strip()

                ollama_result = self.call_ollama(retry_prompt)
                raw_text = ollama_result.get("response", "")

                parsed = json.loads(raw_text)
                parsed_ok = True
                validated_ok = self.is_valid_result(parsed)

        except Exception as e:
            ollama_result = {}
            raw_text = raw_text or str(e)
            parsed = self.fallback_result(raw_text)
            parsed_ok = False
            validated_ok = False

        if not validated_ok:
            parsed = self.fallback_result(raw_text)

        parsed = self.validate_and_fix_result(parsed, comparison_data, apply_forced_route=validated_ok)

        return {
            "ok": True,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "model": self.model_name,
            "parsed_ok": parsed_ok,
            "validated_ok": validated_ok,
            "retry_used": retry_used,
            "result": parsed,
            "raw_text": raw_text,
            "ollama_metrics": {
                "total_duration": ollama_result.get("total_duration"),
                "load_duration": ollama_result.get("load_duration"),
                "prompt_eval_count": ollama_result.get("prompt_eval_count"),
                "eval_count": ollama_result.get("eval_count"),
            },
        }
