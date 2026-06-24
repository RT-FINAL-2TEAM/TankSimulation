import math

def point_to_segment_distance(p: tuple[float, float], a: tuple[float, float], b: tuple[float, float]) -> float:
    """선분 AB와 점 P 사이의 최단 거리를 계산합니다."""
    ab_vec = (b[0] - a[0], b[1] - a[1])
    ap_vec = (p[0] - a[0], p[1] - a[1])
    
    ab_len_sq = ab_vec[0]**2 + ab_vec[1]**2
    if ab_len_sq == 0.0:
        # A와 B가 같은 점일 경우
        return math.sqrt(ap_vec[0]**2 + ap_vec[1]**2)
        
    t = (ap_vec[0] * ab_vec[0] + ap_vec[1] * ab_vec[1]) / ab_len_sq
    t = max(0.0, min(1.0, t))
    
    proj_p = (a[0] + t * ab_vec[0], a[1] + t * ab_vec[1])
    return math.sqrt((p[0] - proj_p[0])**2 + (p[1] - proj_p[1])**2)

def calculate_cross_track_error(trajectory: list[tuple[float, float]], route: list[tuple[float, float]]) -> tuple[float, float]:
    """실제 주행 궤적과 계획된 경로 간의 평균/최대 Cross-Track Error를 반환합니다.
    return: (mean_cte, max_cte)
    """
    if not trajectory or not route:
        return 0.0, 0.0
        
    if len(route) == 1:
        route = [route[0], route[0]]
        
    errors = []
    for p in trajectory:
        # 각 궤적 점에 대해 모든 선분 중 가장 짧은 거리를 찾음
        min_dist = float('inf')
        for i in range(len(route) - 1):
            d = point_to_segment_distance(p, route[i], route[i+1])
            if d < min_dist:
                min_dist = d
        errors.append(min_dist)
        
    if not errors:
        return 0.0, 0.0
        
    mean_cte = sum(errors) / len(errors)
    max_cte = max(errors)
    return mean_cte, max_cte
