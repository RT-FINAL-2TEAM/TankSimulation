import json
import math
import heapq
from collections import deque

# 채널 중심 추종 / 사이드 바이어스 튜닝 상수
# clearance: 셀에서 가장 가까운 장애물까지 거리(m). 이 값보다 가까우면 비용 가산 → 벽에서 멀어짐
CLEARANCE_DESIRED = 5      # 이상적 여유공간(m). 채널 폭이 넓으면 중앙으로 정렬
CLEARANCE_WEIGHT = 0.4     # 여유공간 부족 1m당 추가 비용 (너무 크면 좁은 통로를 회피)
# side bias: 의도한 채널(A=서쪽 / B=동쪽)을 벗어나면 비용 가산 → 중앙 섬 반대편으로 새지 않음
SIDE_TOL = 7.0             # 기준선에서 이만큼은 허용(m)
SIDE_WEIGHT = 2.0          # 기준선 반대편으로 1m 넘어갈 때마다 추가 비용
# 이미 지나친(전차 뒤) 웨이포인트는 버려서 경로가 뒤로 갔다 오는 hook을 막는다.
# (너무 크면 뒤 웨이포인트로 후진 경로가 생김 — 1m만 허용)
WAYPOINT_PASSED_TOL = 1.0  # 전차 z보다 이만큼 이상 뒤(z 작음)인 웨이포인트는 통과한 것으로 보고 제외(m)
# 전차 차폭 보정: 큰 바위 반경에 소폭 버퍼를 더해 차체 충돌 여유를 준다.
# (전역값이라 크게 잡으면 B 코리더가 막히므로 작게; 큰 여유는 루트 웨이포인트로 확보)
TANK_HALF_WIDTH = 1.0      # 바위 반경에 더할 차폭 버퍼(m). 2로 키우면 route B가 막힘
ROCK_RADIUS = 4.0          # 큰 바위 물리 반경(m)

def create_grid(width: int, height: int, resolution: float) -> list[list[int]]:
    """해상도에 맞춘 2차원 빈 격자 맵을 생성합니다."""
    cols = int(width / resolution)
    rows = int(height / resolution)
    return [[0 for _ in range(cols)] for _ in range(rows)]

def add_obstacles(grid: list[list[int]], obstacles: list[dict], res: float, inflate: float):
    """장애물 영역과 안전 반경(inflate)을 격자에 1로 마킹합니다."""
    rows, cols = len(grid), len(grid[0])
    for obs in obstacles:
        x_min = max(0, int((obs['x_min'] - inflate) / res))
        x_max = min(cols - 1, int((obs['x_max'] + inflate) / res))
        z_min = max(0, int((obs['z_min'] - inflate) / res))
        z_max = min(rows - 1, int((obs['z_max'] + inflate) / res))
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
) -> list[list[float]]:
    """A* 이동 비용에 더할 셀별 추가 비용맵을 생성합니다.

    - 채널 중심 비용: 여유공간(clearance)이 부족한 셀(벽 근처)에 비용 가산 → 중앙 정렬
    - 사이드 바이어스: 의도한 채널 반대편(중앙 섬 쪽)으로 새는 셀에 비용 가산
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

def load_static_obstacles_from_map(map_path: str) -> list[dict]:
    """맵 파일에서 고정된 지형지물(Tree, Rock, Wall 등) 위치를 읽어 bbox 형태로 반환합니다."""
    try:
        with open(map_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception:
        return []
    obstacles = []
    for obs in data.get('obstacles', []):
        prefab = obs.get('prefabName', '')
        # 동적/위협 객체는 제외
        if prefab.startswith('Human') or prefab.startswith('Tank') or prefab.startswith('House'):
            continue
        
        pos = obs.get('position', {})
        x, z = pos.get('x', 0.0), pos.get('z', 0.0)
        
        # 크기 추정 (프리팹 이름 기반)
        # 3.5m 반경은 x≈105~110 인접 나무들이 z=15 레벨까지 겹쳐 B루트 통로를 막으므로 2.5m로 축소
        radius = 2.5
        obs_type = 'Tree'
        if 'Rock' in prefab:
            # 큰 바위: 물리 반경 + 전차 반폭만큼 더 띄워 경로 중심선이 차체를 바위 밖으로 유지
            # (경로 중심선이 바위 5m 밖이어도 차폭 2m면 차체가 바위에 닿아 끼이던 문제 해결)
            radius = ROCK_RADIUS + TANK_HALF_WIDTH
            obs_type = 'Rock'
        elif 'Wall' in prefab:
            radius = 2.0
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
    clearance_weight: float = 0.0,
    waypoints_ref: list[tuple[float, float]] = None,
    side: str = None,
) -> list[tuple[float, float]]:
    """시작점에서 목표점까지의 전역 경로를 생성합니다.

    static_obstacles: 맵에서 미리 로드한 정적 장애물 (inflate 없이 적용).
    clearance_weight: >0이면 벽에서 멀어지는(채널 중심) 비용을 추가.
    waypoints_ref/side: 'east'(B)/'west'(A) 채널을 벗어나지 않도록 사이드 바이어스 적용.
    """
    res = 1.0  # 1m 격자
    grid = create_grid(300, 300, res)
    if static_obstacles:
        # inflate=1.0: A* 경로가 나무 중심에서 최소 3.5m(bbox 2.5m + 1.0m 여유) 이격
        # inflate=0으로는 bbox 경계 바로 옆을 지나 실제 주행 편차로 충돌이 발생
        # B루트 z=15 통로(x=110~121 간격 10.6m)에서 gap=3.6m → 1m 격자에서 통과 가능
        add_obstacles(grid, static_obstacles, res, inflate=1.0)
    add_obstacles(grid, obstacles, res, inflate=inflate)

    cost_map = None
    if clearance_weight > 0 or side:
        cost_map = _build_cost_map(grid, res, clearance_weight, waypoints_ref, side)

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
    clearance_weight: float = 0.0,
    side: str = None,
) -> list[tuple[float, float]]:
    """웨이포인트를 순서대로 경유하는 연결 경로를 생성합니다.

    clearance_weight/side: 채널 중심 추종 + 사이드 바이어스. 웨이포인트는 x좌표를 강제하는
    통과점이 아니라 '의도한 채널' 힌트로 작동하며, 막힌 웨이포인트는 직진하지 않고 건너뜁니다.
    """
    # 격자와 비용맵은 세그먼트 간 동일하므로 한 번만 생성해 재사용 (성능: 8x 빠름)
    res = 1.0
    grid = create_grid(300, 300, res)
    if static_obstacles:
        add_obstacles(grid, static_obstacles, res, inflate=1.0)
    add_obstacles(grid, dynamic_obstacles, res, inflate=inflate)

    cost_map = None
    if clearance_weight > 0 or side:
        cost_map = _build_cost_map(grid, res, clearance_weight, waypoints, side)

    full_path: list[tuple[float, float]] = []
    prev = start_pos
    # 현재 위치보다 z축(북쪽)으로 뒤쳐진 웨이포인트는 이미 지나온 것으로 간주하고 버림 (후진/hook 방지).
    # 단, 마지막(목적지) 웨이포인트는 항상 남겨 경로가 목적지까지 이어지게 한다.
    valid_waypoints = [wp for wp in waypoints if wp[1] >= start_pos[1] - WAYPOINT_PASSED_TOL]
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
