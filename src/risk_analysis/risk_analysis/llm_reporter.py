import json
import textwrap
from datetime import datetime

import requests


class LLMReporter:
    def __init__(
        self,
        ollama_url: str = "http://localhost:11434/api/generate",
        model_name: str = "qwen3:0.6b",
        timeout_sec: int = 1800,
    ):
        self.ollama_url = ollama_url
        self.model_name = model_name
        self.timeout_sec = timeout_sec

    def build_prompt(self, comparison_data: dict) -> str:
        return textwrap.dedent(f"""
        /no_think

        너는 전차 정찰 임무의 전술 참모 AI다.

        아래 데이터는 동일한 목표지점으로 가는 A 루트와 B 루트를 정찰한 결과다.
        각 데이터는 센서와 시뮬레이터에서 측정된 관측값이며, 별도의 위험 점수는 제공되지 않는다.

        임무:
        1. A 루트와 B 루트의 위험 요소를 비교한다.
        2. 고정 가중치 계산을 하지 않는다.
        3. 전술적 관점에서 어느 루트가 더 안전한지 판단한다.
        4. 반드시 JSON 형식으로만 출력한다.

        중요 판단 원칙:
        - 적 노출 시간이 긴 루트는 매우 위험하다.
        - max_continuous_visible_time_s가 길수록 지속 노출 위험이 크다.
        - closest_enemy_distance_m가 짧을수록 교전 또는 피탐지 위험이 크다.
        - enemy_count가 많을수록 위험하다.
        - blocked_segment_count가 1 이상이면 이동 지연 또는 매복 위험이 크다.
        - obstacle_count가 많고 blocked_segment_count가 있으면 불리하다.
        - pitch_std_deg, roll_std_deg가 높으면 지형 주행 안정성이 낮다.
        - terrain roughness가 다소 높더라도 적 노출이 짧고 차단 구간이 없으면 더 안전할 수 있다.
        - 단순히 한 항목만 보고 판단하지 말고 전체 위험 맥락을 비교하라.

        엄격한 출력 규칙:
        - 반드시 JSON 객체 하나만 출력한다.
        - 설명 문장, 마크다운, 코드블록을 출력하지 마라.
        - 빈 JSON 객체 {{}}는 실패 응답이다.
        - 절대로 {{}}만 출력하지 마라.
        - 반드시 모든 필드를 채워라.
        - selected_route는 반드시 "A" 또는 "B" 중 하나만 선택한다.
        - risk_level.A와 risk_level.B는 반드시 "low", "medium", "high", "critical" 중 하나만 선택한다.
        - confidence는 반드시 "low", "medium", "high" 중 하나만 선택한다.
        - recommended_behavior.speed_policy는 반드시 "slow", "medium", "fast" 중 하나만 선택한다.
        - used_evidence의 enemy_visible_time_s, obstacle_count, blocked_segment_count는 입력 데이터의 실제 값을 그대로 복사한다.
        - used_evidence의 숫자를 null 또는 0으로 임의 작성하지 마라.
        - 입력에 있는 실제 숫자를 반드시 사용한다.
        - decision_reason과 used_evidence.reason은 selected_route와 모순되면 안 된다.
        - key_risks에는 항목명만 쓰지 말고 실제 위험 내용을 한국어로 설명한다.
        - summary는 한국어 한 문장으로 작성한다.
        - decision_reason은 선택 이유를 한국어로 구체적으로 설명한다.

        출력 JSON의 최상위 키는 반드시 아래 8개를 모두 포함해야 한다:
        selected_route, risk_level, confidence, summary, decision_reason, key_risks, recommended_behavior, used_evidence

        필드 규칙:
        - selected_route: "A" 또는 "B"
        - risk_level.A: "low", "medium", "high", "critical" 중 하나
        - risk_level.B: "low", "medium", "high", "critical" 중 하나
        - confidence: "low", "medium", "high" 중 하나
        - summary: 한국어 한 문장
        - decision_reason: 선택 이유를 한국어로 설명
        - key_risks.A: A 루트의 주요 위험 요인 배열
        - key_risks.B: B 루트의 주요 위험 요인 배열
        - recommended_behavior.speed_policy: "slow", "medium", "fast" 중 하나
        - recommended_behavior.caution_points: 주의점 배열
        - recommended_behavior.tactical_note: 전술 메모
        - used_evidence.A.enemy_visible_time_s: 입력 route_A의 실제 enemy_visible_time_s 값
        - used_evidence.A.obstacle_count: 입력 route_A의 실제 obstacle_count 값
        - used_evidence.A.blocked_segment_count: 입력 route_A의 실제 blocked_segment_count 값
        - used_evidence.B.enemy_visible_time_s: 입력 route_B의 실제 enemy_visible_time_s 값
        - used_evidence.B.obstacle_count: 입력 route_B의 실제 obstacle_count 값
        - used_evidence.B.blocked_segment_count: 입력 route_B의 실제 blocked_segment_count 값

        반드시 아래 JSON 구조만 출력한다:
        {{
          "selected_route": "",
          "risk_level": {{
            "A": "",
            "B": ""
          }},
          "confidence": "",
          "summary": "",
          "decision_reason": "",
          "key_risks": {{
            "A": [],
            "B": []
          }},
          "recommended_behavior": {{
            "speed_policy": "",
            "caution_points": [],
            "tactical_note": ""
          }},
          "used_evidence": {{
            "A": {{
              "enemy_visible_time_s": null,
              "obstacle_count": null,
              "blocked_segment_count": null,
              "reason": ""
            }},
            "B": {{
              "enemy_visible_time_s": null,
              "obstacle_count": null,
              "blocked_segment_count": null,
              "reason": ""
            }}
          }}
        }}

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

    def validate_and_fix_result(self, parsed: dict, comparison_data: dict) -> dict:
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

        # used_evidence의 숫자는 LLM 결과를 믿지 않고 입력 JSON의 실제 값으로 덮어쓴다.
        parsed["used_evidence"] = {
            "A": {
                "enemy_visible_time_s": route_a.get("enemy_visible_time_s"),
                "obstacle_count": route_a.get("obstacle_count"),
                "blocked_segment_count": route_a.get("blocked_segment_count"),
                "reason": old_a.get("reason", ""),
            },
            "B": {
                "enemy_visible_time_s": route_b.get("enemy_visible_time_s"),
                "obstacle_count": route_b.get("obstacle_count"),
                "blocked_segment_count": route_b.get("blocked_segment_count"),
                "reason": old_b.get("reason", ""),
            },
        }

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

        parsed = self.validate_and_fix_result(parsed, comparison_data)

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