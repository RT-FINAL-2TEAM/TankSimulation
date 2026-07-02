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
                 cost_map: list[list[float]] = None) -> list[tuple[int, int]]:
    """A* 알고리즘을 사용해 최단 경로를 탐색합니다.

    cost_map: 셀별 추가 이동 비용(채널 중심/사이드 바이어스). 모두 음이 아니므로
    휴리스틱(직선거리)은 여전히 admissible → 경로 정확성 유지.
    """
    frontier = []
    heapq.heappush(frontier, (0, start))
    came_from = {start: None}
    cost_so_far = {start: 0}

    while frontier:
        _, current = heapq.heappop(frontier)
        if current == goal:
            break

        for next_node in get_neighbors(current, grid):
            # 대각선 이동은 비용 1.414, 직진은 1.0
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
    if clearance_weight > 0 or side or terrain_grid:
        cost_map = _build_cost_map(grid, res, clearance_weight, waypoints_ref, side,
                                   terrain_grid, terrain_weight)

    start_grid = (int(start_pos[0] / res), int(start_pos[1] / res))
    goal_grid = (int(goal_pos[0] / res), int(goal_pos[1] / res))
    # 목표가 막혀 있으면 벽으로 직진하지 않도록 의도한 채널 쪽 free 셀로 스냅
    goal_grid = _snap_goal(grid, goal_grid[0], goal_grid[1], side)

    grid_path = astar_search(grid, start_grid, goal_grid, cost_map=cost_map)
    if not grid_path:
        return []

    smoothed_grid_path = smooth_path(grid, grid_path)

    # 격자 좌표를 다시 실제 좌표(m)로 변환
    real_path = [(p[0] * res, p[1] * res) for p in smoothed_grid_path]
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
    if clearance_weight > 0 or side or terrain_grid:
        cost_map = _build_cost_map(grid, res, clearance_weight, waypoints, side,
                                   terrain_grid, terrain_weight)

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
        grid_path = astar_search(grid, start_grid, goal_grid, cost_map=cost_map)
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
    return full_path
