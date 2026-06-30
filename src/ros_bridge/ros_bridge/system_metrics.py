"""시스템 자원 모니터링(CPU·메모리·GPU) — C2 대시보드 ③ 패널용.

각 함수는 try/except로 graceful: psutil 미설치/ GPU 없음이어도 빈 dict를 반환해
대시보드 빌드가 깨지지 않게 한다. _build_dashboard_payload(백그라운드 리프레셔)에서 호출.
"""

from __future__ import annotations

import subprocess
from typing import Any, Dict

try:
    import psutil  # CPU/메모리
except Exception:  # pragma: no cover - 의존성 미설치 graceful
    psutil = None


def get_cpu_memory_metrics() -> Dict[str, Any]:
    """CPU%·메모리 사용률. psutil 없으면 빈 dict."""
    if psutil is None:
        return {"psutilAvailable": False}
    try:
        vm = psutil.virtual_memory()
        return {
            "psutilAvailable": True,
            "cpuPercent": round(psutil.cpu_percent(interval=None), 1),  # 직전 호출 이후 평균(비차단)
            "memoryPercent": round(vm.percent, 1),
            "memoryUsedMb": round(vm.used / 1024 / 1024, 0),
            "memoryTotalMb": round(vm.total / 1024 / 1024, 0),
        }
    except Exception:
        return {"psutilAvailable": False}


def get_gpu_metrics() -> Dict[str, Any]:
    """GPU 사용률·메모리(nvidia-smi). GPU/드라이버 없으면 gpuAvailable=False."""
    try:
        result = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=utilization.gpu,memory.used,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=2,
        )
        if result.returncode == 0 and result.stdout.strip():
            # 멀티 GPU면 첫 줄만 사용
            line = result.stdout.strip().splitlines()[0]
            util, used, total = (p.strip() for p in line.split(","))
            return {
                "gpuAvailable": True,
                "gpuPercent": float(util),
                "gpuMemoryUsedMb": float(used),
                "gpuMemoryTotalMb": float(total),
            }
    except (FileNotFoundError, subprocess.TimeoutExpired, ValueError, Exception):
        pass
    return {"gpuAvailable": False}


def get_system_metrics() -> Dict[str, Any]:
    """대시보드 payload에 넣을 통합 시스템 메트릭."""
    metrics: Dict[str, Any] = {}
    metrics.update(get_cpu_memory_metrics())
    metrics.update(get_gpu_metrics())
    return metrics
