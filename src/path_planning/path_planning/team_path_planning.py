import json
import math
import heapq
from collections import deque

# 채널 중심 추종 / 사이드 바이어스 튜닝 상수
# clearance: 셀에서 가장 가까운 장애물까지 거리(m). 이 값보다 가까우면 비용 가산 → 벽에서 멀어짐
CLEARANCE_DESIRED = 5      # 이상적 여유공간(m). 채널 폭이 넓으면 중앙으로 정렬
CLEARANCE_WEIGHT = 0.35    # 여유공간 부족 1m당 추가 비용 (너무 크면 좁은 통로를 회피)
# side bias: 의도한 채널(A=서쪽 / B=동쪽)을 벗어나면 비용 가산 → 중앙 섬 반대편으로 새지 않음
SIDE_TOL = 10.0            # 기준선에서 이만큼은 허용(m) - A/B 경로 유지용 허용 폭
SIDE_WEIGHT = 6.0          # 기준선 반대편으로 1m 넘어갈 때마다 추가 비용 - 반대편 경로 튐 방지
# 이미 지나친(전차 뒤) 웨이포인트는 버려서 경로가 뒤로 갔다 오는 hook을 막는다.
# (너무 크면 뒤 웨이포인트로 후진 경로가 생김 — 1m만 허용)
WAYPOINT_PASSED_TOL = 1.0  # 전차 z보다 이만큼 이상 뒤(z 작음)인 웨이포인트는 통과한 것으로 보고 제외(m)
# off-route 복귀: 웨이포인트를 z(북향)로 지났어도 측면(루트축 수직=현 N-S 루트에선 x)으로 이만큼 넘게
# 벗어났으면 '지남'으로 보지 않는다 → 동쪽으로 밀려나도 서쪽 웨이포인트를 유지해 A*가 루트로 되돌아간다.
# (없으면 z만으로 지남 판정 → 측면 이탈 시 웨이포인트 삭제 → 목적지 최단=중앙 숲 관통.)
WAYPOINT_LATERAL_TOL = 18.0
# 전차 차폭 보정: 큰 바위 반경에 소폭 버퍼를 더해 차체 충돌 여유를 준다.
# Known map obstacle은 자율주행 전 이미 알고 있는 hard no-go 정보이므로,
# tree/rock/wall을 너무 얇게 넣으면 A*가 초록점 사이를 가로질러 간다.
TANK_HALF_WIDTH = 2.0      # 전차 폭 4m 기준 반폭 버퍼(m)
ROCK_RADIUS = 4.0          # 큰 바위 물리 반경(m). 실제 폭 8m 기준: 중심 반경 4m
TREE_RADIUS = 4.0          # finalmap Tree hard no-go 반경(m). 과팽창으로 인한 경로 와리가리 완화
DEFAULT_STATIC_INFLATE = 2.0  # static map obstacle A* hard inflation(m). 실제 반경 + 전차 반폭 + 안전여유

def create_grid(width: int, height: int, resolution: float) -> list[list[int]]:
    """해상도에 맞춘 2차원 빈 격자 맵을 생성합니다."""
    cols = int(width / resolution)
    rows = int(height / resolution)
    return [[0 for _ in range(cols)] for _ in range(rows)]

def add_obstacles(grid: list[list[int]], obstacles: list[dict], res: float, inflate: float):
    """장애물 영역과 안전 반경(inflate)을 격자에 1로 마킹합니다."""
    rows, cols = len(grid), len(grid[0])
    for obs in obstacles:
        # dynamic lidar / discovered / static obstacle이 서로 다른 inflate를 쓸 수 있게 한다.
        # _inflate_override가 없으면 함수 인자로 받은 기본 inflate를 사용한다.
        try:
            obs_inflate = float(obs.get('_inflate_override', inflate)) if isinstance(obs, dict) else float(inflate)
        except Exception:
            obs_inflate = float(inflate)
        x_min = max(0, int((obs['x_min'] - obs_inflate) / res))
        x_max = min(cols - 1, int((obs['x_max'] + obs_inflate) / res))
        z_min = max(0, int((obs['z_min'] - obs_inflate) / res))
        z_max = min(rows - 1, int((obs['z_max'] + obs_inflate) / res))
        for z in range(z_min, z_max + 1):
            for x in range(x_min, x_max + 1):
                grid[z][x] = 1

def compute_clearance(grid: list[list[int]], max_r: int) -> list[list[int]]:
    """각 free 셀에서 가장 가까운 장애물까지의 격자 거리(상한 max_r)를 다중 소스 BFS로 계산합니다.

    채널 중심일수록 값이 크고(여유 많음), 벽에 붙을수록 0에 가깝습니다.
    """
    rows, cols = len(grid), len(grid[0])
    # 장애물에서 max_r 이상 떨어진 셀은 max_r로 고정(추가 비용 0)
    dist = [[max_r] * cols for _ in range(rows)]
    dq = deque()
    for z in range(rows):
        row = grid[z]
        drow = dist[z]
        for x in range(cols):
            if row[x] == 1:
                drow[x] = 0
                dq.append((x, z))
    while dq:
        x, z = dq.popleft()
        d = dist[z][x]
        if d >= max_r:
            continue
        nd = d + 1
        for dx, dz in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nx, nz = x + dx, z + dz
            if 0 <= nx < cols and 0 <= nz < rows and dist[nz][nx] > nd:
                dist[nz][nx] = nd
                dq.append((nx, nz))
    return dist


def _x_ref_at(z_real: float, waypoints: list[tuple[float, float]]):
    """웨이포인트 폴리라인을 z기준으로 보간해 기준 x를 반환합니다(사이드 바이어스용)."""
    if not waypoints:
        return None
    pts = sorted(waypoints, key=lambda p: p[1])
    if z_real <= pts[0][1]:
        return pts[0][0]
    if z_real >= pts[-1][1]:
        return pts[-1][0]
    for i in range(len(pts) - 1):
        z0, z1 = pts[i][1], pts[i + 1][1]
        if z0 <= z_real <= z1:
            if z1 == z0:
                return pts[i][0]
            t = (z_real - z0) / (z1 - z0)
            return pts[i][0] + t * (pts[i + 1][0] - pts[i][0])
    return pts[-1][0]


def _build_cost_map(
    grid: list[list[int]], res: float, clearance_weight: float,
    waypoints: list[tuple[float, float]] = None, side: str = None,
    terrain_grid: dict = None, terrain_weight: float = 0.0,
    semantic_risk_sources: list[dict] | None = None,
    semantic_risk_scores: dict[str, float] | None = None,
    semantic_risk_radii: dict[str, float] | None = None,
    semantic_risk_weight: float = 0.0,
    semantic_risk_radius_scale: float = 1.0,
) -> list[list[float]]:
    """A* 이동 비용에 더할 셀별 추가 비용맵을 생성합니다.

    - 채널 중심 비용: 여유공간(clearance)이 부족한 셀(벽 근처)에 비용 가산 → 중앙 정렬
    - 사이드 바이어스: 의도한 채널 반대편(중앙 섬 쪽)으로 새는 셀에 비용 가산
    - 지형 비용(게이트형): terrain_grid가 주어질 때만 적용. 거친(고도차 큰) 셀일수록 비용 가산
      → 시나리오2에서 험지 회피. 정찰엔 terrain_grid=None이라 무영향(기존 동작 동일).
      terrain_grid = {(ix, iy): roughness(dz/m)} — A* 격자 인덱스(ix=열=map x, iy=행=map y) 기준.
    """
    rows, cols = len(grid), len(grid[0])
    cost = [[0.0] * cols for _ in range(rows)]

    if clearance_weight > 0:
        dist = compute_clearance(grid, max_r=CLEARANCE_DESIRED)
        for z in range(rows):
            crow, drow, grow = cost[z], dist[z], grid[z]
            for x in range(cols):
                if grow[x] == 0:
                    crow[x] += clearance_weight * (CLEARANCE_DESIRED - drow[x])

    if side in ('east', 'west') and waypoints:
        # 행마다 기준 x를 한 번만 보간
        for z in range(rows):
            xref = _x_ref_at(z * res, waypoints)
            if xref is None:
                continue
            crow, grow = cost[z], grid[z]
            if side == 'east':
                bound = xref - SIDE_TOL  # 이보다 서쪽이면 벌점
                for x in range(cols):
                    if grow[x] == 0:
                        over = bound - x * res
                        if over > 0:
                            crow[x] += SIDE_WEIGHT * over
            else:  # west
                bound = xref + SIDE_TOL  # 이보다 동쪽이면 벌점
                for x in range(cols):
                    if grow[x] == 0:
                        over = x * res - bound
                        if over > 0:
                            crow[x] += SIDE_WEIGHT * over

    # 지형 비용(게이트형): terrain_grid 있을 때만. 희소 dict라 셀 단위로만 순회.
    if terrain_grid and terrain_weight > 0:
        for (ix, iy), rough in terrain_grid.items():
            if 0 <= iy < rows and 0 <= ix < cols and grid[iy][ix] == 0:
                cost[iy][ix] += terrain_weight * float(rough)

    # Semantic risk cost: tank/house/car/tent/rock처럼 의미가 확정된 발견 객체 주변에
    # class별 soft cost를 추가한다. 충돌 영역 자체는 hard obstacle로 유지된다.
    _add_semantic_risk_cost(
        cost, grid, res, semantic_risk_sources, semantic_risk_scores, semantic_risk_radii,
        semantic_risk_weight, semantic_risk_radius_scale
    )
    return cost


def _snap_goal(grid: list[list[int]], gx: int, gz: int, prefer_side: str = None,
               window: int = 16) -> tuple[int, int]:
    """목표 셀이 막혀 있으면 주변 free 셀로 이동합니다(의도한 채널 방향 우선).

    웨이포인트가 라이다/정적 장애물로 막힌 경우(예: 중앙 섬 가장자리), 벽으로 직진하는 대신
    가까우면서 의도한 채널(B=동/A=서) 쪽인 통과 가능 셀로 스냅합니다.
    """
    rows, cols = len(grid), len(grid[0])
    if 0 <= gx < cols and 0 <= gz < rows and grid[gz][gx] == 0:
        return (gx, gz)
    best = None
    best_score = -1e9
    for dz in range(-window, window + 1):
        for dx in range(-window, window + 1):
            nx, nz = gx + dx, gz + dz
            if 0 <= nx < cols and 0 <= nz < rows and grid[nz][nx] == 0:
                score = -math.hypot(dx, dz)  # 가까울수록 우선
                if prefer_side == 'east' and dx > 0:
                    score += 4.0
                elif prefer_side == 'west' and dx < 0:
                    score += 4.0
                if score > best_score:
                    best_score = score
                    best = (nx, nz)
    return best if best else (gx, gz)


def heuristic(a: tuple[int, int], b: tuple[int, int]) -> float:
    """A* 휴리스틱(유클리드 거리)을 계산합니다."""
    return math.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)

def get_neighbors(node: tuple[int, int], grid: list[list[int]]) -> list[tuple[int, int]]:
    """현재 노드의 8방향 이동 가능한 이웃을 반환합니다."""
    neighbors = []
    directions = [
        (0, 1), (1, 0), (0, -1), (-1, 0),
        (1, 1), (1, -1), (-1, 1), (-1, -1)
    ]
    rows, cols = len(grid), len(grid[0])
    for dx, dy in directions:
        nx, ny = node[0] + dx, node[1] + dy
        if 0 <= nx < cols and 0 <= ny < rows:
            if grid[ny][nx] == 0:
                neighbors.append((nx, ny))
    return neighbors

def astar_search(grid: list[list[int]], start: tuple[int, int], goal: tuple[int, int],
                 cost_map: list[list[float]] = None,
                 heading_change_weight: float = 0.0) -> list[tuple[int, int]]:
    """A* 알고리즘을 사용해 경로를 탐색합니다.

    cost_map: 셀별 추가 이동 비용(채널 중심/사이드 바이어스/semantic risk/terrain).
    heading_change_weight > 0이면 state를 (x, y, dir_idx)로 확장해 방향 변화 비용을 추가한다.
    완전한 Hybrid A*는 아니지만, 2D A*의 급격한 꺾임을 줄이는 theta-aware-lite 단계다.
    """
    directions = [
        (0, 1), (1, 0), (0, -1), (-1, 0),
        (1, 1), (1, -1), (-1, 1), (-1, -1)
    ]

    if heading_change_weight <= 1e-9:
        frontier = []
        heapq.heappush(frontier, (0, start))
        came_from = {start: None}
        cost_so_far = {start: 0}
        while frontier:
            _, current = heapq.heappop(frontier)
            if current == goal:
                break
            for next_node in get_neighbors(current, grid):
                move_cost = 1.414 if next_node[0] != current[0] and next_node[1] != current[1] else 1.0
                if cost_map is not None:
                    move_cost += cost_map[next_node[1]][next_node[0]]
                new_cost = cost_so_far[current] + move_cost
                if next_node not in cost_so_far or new_cost < cost_so_far[next_node]:
                    cost_so_far[next_node] = new_cost
                    priority = new_cost + heuristic(next_node, goal)
                    heapq.heappush(frontier, (priority, next_node))
                    came_from[next_node] = current
        return reconstruct_path(came_from, start, goal)

    rows, cols = len(grid), len(grid[0])
    start_state = (start[0], start[1], -1)  # -1 = 아직 진행방향 미정
    frontier: list[tuple[float, tuple[int, int, int]]] = []
    heapq.heappush(frontier, (heuristic(start, goal), start_state))
    came_from: dict[tuple[int, int, int], tuple[int, int, int] | None] = {start_state: None}
    cost_so_far: dict[tuple[int, int, int], float] = {start_state: 0.0}
    goal_state = None

    while frontier:
        _, current = heapq.heappop(frontier)
        x, y, prev_dir = current
        if (x, y) == goal:
            goal_state = current
            break
        for dir_idx, (dx, dy) in enumerate(directions):
            nx, ny = x + dx, y + dy
            if not (0 <= nx < cols and 0 <= ny < rows):
                continue
            if grid[ny][nx] != 0:
                continue
            move_cost = 1.414 if dx != 0 and dy != 0 else 1.0
            if cost_map is not None:
                move_cost += cost_map[ny][nx]
            if prev_dir >= 0:
                turn_steps = abs(dir_idx - prev_dir)
                turn_steps = min(turn_steps, 8 - turn_steps)
                move_cost += heading_change_weight * float(turn_steps)
            next_state = (nx, ny, dir_idx)
            new_cost = cost_so_far[current] + move_cost
            if next_state not in cost_so_far or new_cost < cost_so_far[next_state]:
                cost_so_far[next_state] = new_cost
                priority = new_cost + heuristic((nx, ny), goal)
                heapq.heappush(frontier, (priority, next_state))
                came_from[next_state] = current

    if goal_state is None:
        return []
    state_path = []
    cur = goal_state
    while cur is not None:
        state_path.append(cur)
        cur = came_from.get(cur)
    state_path.reverse()
    return [(x, y) for x, y, _ in state_path]

def reconstruct_path(came_from: dict, start: tuple[int, int], goal: tuple[int, int]) -> list[tuple[int, int]]:
    """찾아낸 경로 딕셔너리를 역추적하여 리스트로 만듭니다."""
    if goal not in came_from:
        return []
    
    current = goal
    path = []
    while current != start:
        path.append(current)
        current = came_from[current]
    path.append(start)
    path.reverse()
    return path

def has_line_of_sight(grid: list[list[int]], p1: tuple[int, int], p2: tuple[int, int]) -> bool:
    """두 지점 사이에 장애물이 없는지(직선 가시성) 확인합니다."""
    x0, y0 = p1
    x1, y1 = p2
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    x_inc = 1 if x0 < x1 else -1
    y_inc = 1 if y0 < y1 else -1
    err = dx - dy

    while True:
        if grid[y0][x0] == 1:
            return False
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x0 += x_inc
        if e2 < dx:
            err += dx
            y0 += y_inc
            
    return True

def smooth_path(grid: list[list[int]], path: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """경로의 불필요한 웨이포인트를 제거하여 평활화합니다."""
    if len(path) <= 2:
        return path
        
    smoothed = [path[0]]
    current_idx = 0
    
    while current_idx < len(path) - 1:
        furthest_idx = current_idx + 1
        for i in range(current_idx + 2, len(path)):
            if has_line_of_sight(grid, path[current_idx], path[i]):
                furthest_idx = i
            else:
                break
        smoothed.append(path[furthest_idx])
        current_idx = furthest_idx
        
    return smoothed



def _path_dist(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(float(b[0]) - float(a[0]), float(b[1]) - float(a[1]))


def _unit_vec(a: tuple[float, float], b: tuple[float, float]) -> tuple[float, float]:
    d = _path_dist(a, b)
    if d <= 1.0e-9:
        return (0.0, 0.0)
    return ((float(b[0]) - float(a[0])) / d, (float(b[1]) - float(a[1])) / d)


def _turn_angle_deg(a: tuple[float, float], b: tuple[float, float], c: tuple[float, float]) -> float:
    """연속 3점의 진행방향 변화각(deg)을 반환한다. 직선=0, 직각=90."""
    v1 = _unit_vec(a, b)
    v2 = _unit_vec(b, c)
    dot = max(-1.0, min(1.0, v1[0] * v2[0] + v1[1] * v2[1]))
    return math.degrees(math.acos(dot))


def _append_densified(out: list[tuple[float, float]], p: tuple[float, float], spacing: float) -> None:
    """마지막 점에서 p까지 spacing 간격으로 보간해 append한다."""
    if not out:
        out.append((float(p[0]), float(p[1])))
        return
    last = out[-1]
    dist = _path_dist(last, p)
    if dist <= 1.0e-6:
        return
    n = max(1, int(math.ceil(dist / max(0.2, spacing))))
    for i in range(1, n + 1):
        t = i / n
        out.append((last[0] + (float(p[0]) - last[0]) * t, last[1] + (float(p[1]) - last[1]) * t))


def _curve_is_free(
    grid: list[list[int]],
    curve: list[tuple[float, float]],
    res: float,
    margin_m: float,
) -> bool:
    """곡선 샘플들이 inflated grid 안에서 충돌 없는지 검사한다."""
    if not grid or not grid[0]:
        return True
    rows, cols = len(grid), len(grid[0])
    margin_cells = max(0, int(math.ceil(max(0.0, margin_m) / max(1.0e-6, res))))
    for x_m, z_m in curve:
        ix = int(round(float(x_m) / res))
        iz = int(round(float(z_m) / res))
        if ix < 0 or iz < 0 or ix >= cols or iz >= rows:
            return False
        for dz in range(-margin_cells, margin_cells + 1):
            zz = iz + dz
            if zz < 0 or zz >= rows:
                return False
            for dx in range(-margin_cells, margin_cells + 1):
                xx = ix + dx
                if xx < 0 or xx >= cols:
                    return False
                if grid[zz][xx] != 0:
                    return False
    return True


def _quadratic_bezier(
    p0: tuple[float, float],
    p1: tuple[float, float],
    p2: tuple[float, float],
    spacing: float,
) -> list[tuple[float, float]]:
    approx_len = _path_dist(p0, p1) + _path_dist(p1, p2)
    n = max(4, int(math.ceil(approx_len / max(0.2, spacing))))
    pts: list[tuple[float, float]] = []
    for i in range(n + 1):
        t = i / n
        u = 1.0 - t
        x = u * u * p0[0] + 2.0 * u * t * p1[0] + t * t * p2[0]
        z = u * u * p0[1] + 2.0 * u * t * p1[1] + t * t * p2[1]
        pts.append((x, z))
    return pts


def curvature_smooth_path(
    path: list[tuple[float, float]],
    grid: list[list[int]],
    res: float = 1.0,
    enabled: bool = True,
    min_turn_radius_m: float = 7.0,
    max_corner_angle_deg: float = 25.0,
    point_spacing_m: float = 1.0,
    collision_check_margin_m: float = 1.0,
) -> list[tuple[float, float]]:
    """A* polyline corner를 전차가 선회 주행 가능한 곡선으로 완화한다.

    - A*는 장애물 없는 corridor를 찾고, 이 함수는 그 corridor 안에서 corner를 둥글게 만든다.
    - 각 corner는 quadratic Bezier로 대체한다. 곡선 샘플이 inflated grid와 충돌하면 해당 corner는 원래 polyline으로 둔다.
    - 완전한 Hybrid A*는 아니지만, 정지 후 yaw pivot을 줄이는 post-smoothing 단계다.
    """
    if not enabled or len(path) < 3:
        return path
    spacing = max(0.3, float(point_spacing_m))
    radius = max(0.5, float(min_turn_radius_m))
    threshold = max(1.0, float(max_corner_angle_deg))
    out: list[tuple[float, float]] = [(float(path[0][0]), float(path[0][1]))]
    i = 1
    while i < len(path) - 1:
        prev = (float(path[i - 1][0]), float(path[i - 1][1]))
        cur = (float(path[i][0]), float(path[i][1]))
        nxt = (float(path[i + 1][0]), float(path[i + 1][1]))
        len_in = _path_dist(prev, cur)
        len_out = _path_dist(cur, nxt)
        angle = _turn_angle_deg(prev, cur, nxt)
        if angle < threshold or len_in < spacing * 1.5 or len_out < spacing * 1.5:
            _append_densified(out, cur, spacing)
            i += 1
            continue
        # 회전반경 R의 원호 tangent 길이를 근사하되, segment 절반을 넘지 않도록 제한한다.
        cut = radius * math.tan(math.radians(min(angle, 135.0)) * 0.5)
        cut = min(cut, len_in * 0.45, len_out * 0.45)
        if cut < spacing:
            _append_densified(out, cur, spacing)
            i += 1
            continue
        vin = _unit_vec(prev, cur)
        vout = _unit_vec(cur, nxt)
        tangent_in = (cur[0] - vin[0] * cut, cur[1] - vin[1] * cut)
        tangent_out = (cur[0] + vout[0] * cut, cur[1] + vout[1] * cut)
        curve = _quadratic_bezier(tangent_in, cur, tangent_out, spacing)
        if _curve_is_free(grid, curve, res, collision_check_margin_m):
            _append_densified(out, tangent_in, spacing)
            for p in curve[1:]:
                _append_densified(out, p, spacing)
        else:
            # 곡선이 obstacle inflation을 침범하면 기존 corner를 유지한다.
            _append_densified(out, cur, spacing)
        i += 1
    _append_densified(out, (float(path[-1][0]), float(path[-1][1])), spacing)
    return out

# 정찰(recon) 발견객체 class별 A* 회피 반경(m). 시나리오2 합본맵(scenario2_map.map)에서 사용.
# tank 포함 — 발견 객체는 종류 불문 전부 회피(사용자 결정). person/human은 제외(저신뢰·설계상 제거).
# tank는 여기서 회피 장애물이면서, 생성기가 별도 targets 목록에도 위치를 남겨 차후 교전 로직이 쓴다.
DISCOVERED_CLASS_RADIUS = {
    'rock': ROCK_RADIUS + TANK_HALF_WIDTH,  # 5.0 (finalmap 바위와 동일 정책)
    'car': 3.0,
    'house': 6.0,
    'tent': 2.5,
    'tank': 4.0,                            # 차체 + 버퍼
}


# Semantic risk layer: discovered object를 hard obstacle로만 보지 않고,
# class별 위험 반경 안에 soft cost를 추가한다. 실제 충돌 금지 영역은 기존
# discovered/class radius + inflation이 담당하고, 이 레이어는 "가능하면 멀리 우회" 용도다.
DEFAULT_SEMANTIC_RISK_SCORES = {
    'tank': 100.0,
    'house': 50.0,
    'car': 25.0,
    'tent': 15.0,
    'rock': 10.0,
    'unknown': 5.0,
}
DEFAULT_SEMANTIC_RISK_RADII = {
    'tank': 25.0,
    'house': 18.0,
    'car': 10.0,
    'tent': 8.0,
    'rock': 6.0,
    'unknown': 5.0,
}


def _bbox_center_for_cost(obs: dict) -> tuple[float, float]:
    return (
        0.5 * (float(obs.get('x_min', 0.0)) + float(obs.get('x_max', 0.0))),
        0.5 * (float(obs.get('z_min', 0.0)) + float(obs.get('z_max', 0.0))),
    )


def _add_semantic_risk_cost(
    cost: list[list[float]],
    grid: list[list[int]],
    res: float,
    sources: list[dict] | None,
    scores: dict[str, float] | None,
    radii: dict[str, float] | None,
    weight: float,
    radius_scale: float,
) -> None:
    """class별 semantic risk를 A* soft cost에 추가한다.

    - sources는 discovered bbox 리스트를 사용한다.
    - hard obstacle은 그대로 막고, 그 주변 위험 반경 안의 free cell에만 추가 비용을 준다.
    - 비용은 중심에서 멀어질수록 2차 감쇠한다.
    """
    if not sources or weight <= 0.0:
        return
    rows, cols = len(grid), len(grid[0])
    scores = scores or DEFAULT_SEMANTIC_RISK_SCORES
    radii = radii or DEFAULT_SEMANTIC_RISK_RADII
    radius_scale = max(0.05, float(radius_scale))
    for obs in sources:
        if not isinstance(obs, dict):
            continue
        cls = str(obs.get('class_name', obs.get('type', 'unknown'))).strip().lower() or 'unknown'
        score = float(scores.get(cls, scores.get('unknown', 0.0)))
        radius_m = float(radii.get(cls, radii.get('unknown', 0.0))) * radius_scale
        if score <= 0.0 or radius_m <= 0.0:
            continue
        cx, cz = _bbox_center_for_cost(obs)
        ix0 = max(0, int((cx - radius_m) / res))
        ix1 = min(cols - 1, int((cx + radius_m) / res))
        iz0 = max(0, int((cz - radius_m) / res))
        iz1 = min(rows - 1, int((cz + radius_m) / res))
        for iz in range(iz0, iz1 + 1):
            row_cost = cost[iz]
            row_grid = grid[iz]
            z_m = iz * res
            for ix in range(ix0, ix1 + 1):
                if row_grid[ix] != 0:
                    continue
                x_m = ix * res
                d = math.hypot(x_m - cx, z_m - cz)
                if d > radius_m:
                    continue
                falloff = 1.0 - (d / radius_m)
                row_cost[ix] += weight * score * falloff * falloff


def load_static_obstacles_from_map(map_path: str) -> list[dict]:
    """맵 파일에서 정적 장애물(나무/바위/벽 + 정찰 발견객체)을 읽어 bbox 형태로 반환합니다.

    - finalmap 네이티브(Tree/Rock/Wall): 기존 정책 유지 — 동적/위협 Human/Tank/House는 제외.
    - 정찰 발견객체(`detected_<class>_NNNN` + metadata.class_name): rock/car/house/tent/tank를
      종류 불문 전부 A* 회피 장애물로 포함(class별 반경). person/human만 제외.
      → finalmap 동작 불변(finalmap엔 Tree/Rock/Wall만 존재), 시나리오2 합본맵에서만 발견객체가 추가됨.
    """
    try:
        with open(map_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception:
        return []
    obstacles = []
    for obs in data.get('obstacles', []):
        prefab = str(obs.get('prefabName', ''))
        meta = obs.get('metadata', {}) or {}
        pos = obs.get('position', {})
        x, z = pos.get('x', 0.0), pos.get('z', 0.0)
        # 시나리오2 발견객체(detected_*/metadata.class_name)는 class별 반경으로, finalmap 네이티브는
        # 팀원 fix/control 반경(Tree=TREE_RADIUS, Wall=3.0)으로. 둘을 분기 처리.
        is_discovered = prefab.startswith('detected_') or bool(meta.get('class_name'))
        if is_discovered:
            # 정찰 발견객체: metadata.class_name 우선, 없으면 prefab(detected_<class>_NNNN)에서 추출
            cls = meta.get('class_name')
            if not cls and '_' in prefab:
                parts = prefab.split('_')
                if len(parts) >= 2:
                    cls = parts[1]
            cls = str(cls or '').strip().lower()
            if cls in ('', 'person', 'human'):
                continue  # 저신뢰/설계상 제외
            radius = DISCOVERED_CLASS_RADIUS.get(cls, 3.0)
            obs_type = cls
        else:
            # finalmap 네이티브: 동적/위협 객체(Human/Tank/House)는 제외(정찰 clean-base 원칙)
            if prefab.startswith('Human') or prefab.startswith('Tank') or prefab.startswith('House'):
                continue
            # Tree는 자율주행 전 이미 아는 금지 장애물 → TREE_RADIUS(팀원)로 크게(초록점 밀집 가로지름 방지)
            radius = TREE_RADIUS
            obs_type = 'Tree'
            if 'Rock' in prefab:
                # 큰 바위: 물리 반경 + 전차 반폭만큼 더 띄워 경로 중심선이 차체를 바위 밖으로 유지
                radius = ROCK_RADIUS + TANK_HALF_WIDTH
                obs_type = 'Rock'
            elif 'Wall' in prefab:
                radius = 3.0  # Wall류 static obstacle 반경 +1m(팀원)
                obs_type = 'Wall'

        obstacles.append({
            'type': obs_type,
            'x_min': x - radius,
            'x_max': x + radius,
            'z_min': z - radius,
            'z_max': z + radius,
        })
    return obstacles


def plan_global_path(
    start_pos: tuple[float, float],
    goal_pos: tuple[float, float],
    obstacles: list[dict],
    inflate: float = 3.0,
    static_obstacles: list[dict] = None,
    static_inflate: float = DEFAULT_STATIC_INFLATE,
    clearance_weight: float = 0.0,
    waypoints_ref: list[tuple[float, float]] = None,
    side: str = None,
    terrain_grid: dict = None,
    terrain_weight: float = 0.0,
    semantic_risk_sources: list[dict] | None = None,
    semantic_risk_scores: dict[str, float] | None = None,
    semantic_risk_radii: dict[str, float] | None = None,
    semantic_risk_weight: float = 0.0,
    semantic_risk_radius_scale: float = 1.0,
    heading_change_weight: float = 0.0,
    enable_curvature_smoothing: bool = False,
    curvature_min_turn_radius_m: float = 7.0,
    curvature_max_corner_angle_deg: float = 25.0,
    curvature_point_spacing_m: float = 1.0,
    curvature_collision_check_margin_m: float = 1.0,
) -> list[tuple[float, float]]:
    """시작점에서 목표점까지의 전역 경로를 생성합니다.

    static_obstacles: 맵에서 미리 로드한 정적 장애물.
    static_inflate: 이미 알고 있는 static map obstacle을 hard no-go로 부풀리는 반경.
    clearance_weight: >0이면 벽에서 멀어지는(채널 중심) 비용을 추가.
    waypoints_ref/side: 'east'(B)/'west'(A) 채널을 벗어나지 않도록 사이드 바이어스 적용.
    terrain_grid/terrain_weight: 지형 거칠기 비용(게이트형, 시나리오2). None이면 무영향.
    """
    res = 1.0  # 1m 격자
    grid = create_grid(300, 300, res)
    if static_obstacles:
        # Known map obstacle은 APF 회피 대상이 아니라 A*에서 아예 못 지나가야 하는 hard no-go다.
        add_obstacles(grid, static_obstacles, res, inflate=static_inflate)
    add_obstacles(grid, obstacles, res, inflate=inflate)

    cost_map = None
    if clearance_weight > 0 or side or terrain_grid or semantic_risk_weight > 0:
        cost_map = _build_cost_map(
            grid, res, clearance_weight, waypoints_ref, side, terrain_grid, terrain_weight,
            semantic_risk_sources, semantic_risk_scores, semantic_risk_radii,
            semantic_risk_weight, semantic_risk_radius_scale
        )

    start_grid = (int(start_pos[0] / res), int(start_pos[1] / res))
    goal_grid = (int(goal_pos[0] / res), int(goal_pos[1] / res))
    # 목표가 막혀 있으면 벽으로 직진하지 않도록 의도한 채널 쪽 free 셀로 스냅
    goal_grid = _snap_goal(grid, goal_grid[0], goal_grid[1], side)

    grid_path = astar_search(
        grid, start_grid, goal_grid, cost_map=cost_map,
        heading_change_weight=heading_change_weight
    )
    if not grid_path:
        return []

    smoothed_grid_path = smooth_path(grid, grid_path)

    # 격자 좌표를 다시 실제 좌표(m)로 변환
    real_path = [(p[0] * res, p[1] * res) for p in smoothed_grid_path]
    real_path = curvature_smooth_path(
        real_path, grid, res=res, enabled=enable_curvature_smoothing,
        min_turn_radius_m=curvature_min_turn_radius_m,
        max_corner_angle_deg=curvature_max_corner_angle_deg,
        point_spacing_m=curvature_point_spacing_m,
        collision_check_margin_m=curvature_collision_check_margin_m,
    )
    return real_path


def plan_path_through_waypoints(
    start_pos: tuple[float, float],
    waypoints: list[tuple[float, float]],
    dynamic_obstacles: list[dict],
    static_obstacles: list[dict] = None,
    inflate: float = 3.0,
    static_inflate: float = DEFAULT_STATIC_INFLATE,
    clearance_weight: float = 0.0,
    side: str = None,
    terrain_grid: dict = None,
    terrain_weight: float = 0.0,
    semantic_risk_sources: list[dict] | None = None,
    semantic_risk_scores: dict[str, float] | None = None,
    semantic_risk_radii: dict[str, float] | None = None,
    semantic_risk_weight: float = 0.0,
    semantic_risk_radius_scale: float = 1.0,
    heading_change_weight: float = 0.0,
    enable_curvature_smoothing: bool = False,
    curvature_min_turn_radius_m: float = 7.0,
    curvature_max_corner_angle_deg: float = 25.0,
    curvature_point_spacing_m: float = 1.0,
    curvature_collision_check_margin_m: float = 1.0,
) -> list[tuple[float, float]]:
    """웨이포인트를 순서대로 경유하는 연결 경로를 생성합니다.

    clearance_weight/side: 채널 중심 추종 + 사이드 바이어스. 웨이포인트는 x좌표를 강제하는
    통과점이 아니라 '의도한 채널' 힌트로 작동하며, 막힌 웨이포인트는 직진하지 않고 건너뜁니다.
    terrain_grid/terrain_weight: 지형 거칠기 비용(게이트형, 시나리오2). None이면 무영향.
    """
    # 격자와 비용맵은 세그먼트 간 동일하므로 한 번만 생성해 재사용 (성능: 8x 빠름)
    res = 1.0
    grid = create_grid(300, 300, res)
    if static_obstacles:
        add_obstacles(grid, static_obstacles, res, inflate=static_inflate)
    add_obstacles(grid, dynamic_obstacles, res, inflate=inflate)

    cost_map = None
    if clearance_weight > 0 or side or terrain_grid or semantic_risk_weight > 0:
        cost_map = _build_cost_map(
            grid, res, clearance_weight, waypoints, side, terrain_grid, terrain_weight,
            semantic_risk_sources, semantic_risk_scores, semantic_risk_radii,
            semantic_risk_weight, semantic_risk_radius_scale
        )

    full_path: list[tuple[float, float]] = []
    prev = start_pos
    # 웨이포인트 '지남' 판정 = z(북향) 기준으로 이미 뒤쪽에 있는 checkpoint는 제거한다.
    # 장애물 회피로 x가 크게 벌어진 상태에서도 지난 checkpoint를 유지하면 A*가 뒤로 돌아가는
    # 경로를 생성하므로 lateral 조건을 제거했다. 마지막(목적지)은 항상 남긴다.
    def _wp_passed(wp):
        return wp[1] < start_pos[1] - WAYPOINT_PASSED_TOL
    valid_waypoints = [wp for wp in waypoints if not _wp_passed(wp)]
    if waypoints and (not valid_waypoints or valid_waypoints[-1] is not waypoints[-1]):
        valid_waypoints.append(waypoints[-1])
    for wp in valid_waypoints:
        start_grid = (int(prev[0] / res), int(prev[1] / res))
        goal_grid = _snap_goal(grid, int(wp[0] / res), int(wp[1] / res), side)
        grid_path = astar_search(
        grid, start_grid, goal_grid, cost_map=cost_map,
        heading_change_weight=heading_change_weight
    )
        if not grid_path:
            # 경로 생성 실패: 벽으로 직진하지 않도록 이 웨이포인트는 건너뜀
            # (다음 웨이포인트로 이어가며, 최종적으로 목적지까지 연결을 시도)
            continue
        seg = [(p[0] * res, p[1] * res) for p in smooth_path(grid, grid_path)]
        if full_path:
            full_path.extend(seg[1:])
        else:
            full_path = list(seg)
        # 실제 도달 지점(스냅된 목표일 수 있음)으로 갱신
        prev = seg[-1]
    full_path = curvature_smooth_path(
        full_path, grid, res=res, enabled=enable_curvature_smoothing,
        min_turn_radius_m=curvature_min_turn_radius_m,
        max_corner_angle_deg=curvature_max_corner_angle_deg,
        point_spacing_m=curvature_point_spacing_m,
        collision_check_margin_m=curvature_collision_check_margin_m,
    )
    return full_path
