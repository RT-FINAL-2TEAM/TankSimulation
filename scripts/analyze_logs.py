import os
import json
import glob
import sys
import math
import argparse
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# 로컬 패키지 import를 위해 src 경로 추가
sys.path.append(str(PROJECT_ROOT / "src" / "lidar"))
from lidar.perception_utils import filter_ground_points

def create_report_dir(session_name):
    base_dir = PROJECT_ROOT / "reports"
    report_dir = base_dir / session_name
    report_dir.mkdir(parents=True, exist_ok=True)
    return report_dir

def parse_jsonl(filepath):
    data = []
    if not os.path.exists(filepath):
        return data
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data.append(json.loads(line.strip()))
            except json.JSONDecodeError:
                continue
    return data

def analyze_and_plot(logs_dir=None, map_name="알 수 없음 (시뮬레이터 제공 안됨)", target_mode="latest"):
    if logs_dir is None:
        logs_dir = str(PROJECT_ROOT / "tank_logs")
    logs_path = Path(logs_dir)
    if not logs_path.exists():
        print(f"로그 폴더를 찾을 수 없습니다: {logs_dir}")
        return
        
    session_dirs = [d for d in logs_path.iterdir() if d.is_dir() and d.name.startswith("session_")]
    if not session_dirs:
        print(f"분석할 세션 폴더(session_*)가 {logs_dir} 에 없습니다.")
        return
        
    if target_mode == "auto":
        session_dirs = [d for d in session_dirs if "session_auto_" in d.name]
    elif target_mode == "monitor":
        session_dirs = [d for d in session_dirs if "session_monitor_" in d.name]
        
    if not session_dirs:
        print(f"조건(mode={target_mode})에 맞는 세션 폴더가 없습니다.")
        return
        
    latest_session = max(session_dirs, key=lambda x: x.stat().st_mtime)
    print(f"가장 최근 세션 분석 시작: {latest_session.name}")
    
    is_auto = "session_auto_" in latest_session.name
    
    report_dir = create_report_dir(latest_session.name)
    
    info_file = latest_session / "info.jsonl"
    action_file = latest_session / "get_action.jsonl"
    obstacles_file = latest_session / "obstacles.jsonl"
    fused_file = latest_session / "fused.jsonl"
    destination_file = latest_session / "destination.jsonl"
    collision_file = latest_session / "collision.jsonl"
    
    info_data = parse_jsonl(info_file)
    action_data = parse_jsonl(action_file)
    obstacles_data = parse_jsonl(obstacles_file)
    fused_data = parse_jsonl(fused_file)
    dest_data = parse_jsonl(destination_file)
    collision_data = parse_jsonl(collision_file)
    
    summary_md = f"# 주행 분석 결과 보고서 ({'Auto 모드' if is_auto else 'Monitor 모드'})\n\n"
    summary_md += f"- 생성일시: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
    summary_md += f"- 분석 대상 로그 세션: `{latest_session.name}`\n"
    summary_md += f"- 사용된 맵: **{map_name}**\n"
    summary_md += f"- info 로그 개수: {len(info_data)}\n"
    summary_md += f"- action 로그 개수: {len(action_data)}\n\n"
    
    x_vals, y_vals = [], []
    lidar_x, lidar_y = [], []
    raw_lidar_x, raw_lidar_y = [], []
    
    if info_data:
        for row in info_data:
            if 'data' in row and isinstance(row['data'], dict):
                # 경로 추출
                if 'playerPos' in row['data']:
                    pos = row['data']['playerPos']
                    if 'x' in pos and 'z' in pos:
                        x_vals.append(pos['x'])
                        y_vals.append(pos['z'])
                        
                # LiDAR 포인트 추출 (장애물에 맞은 점들)
                if 'lidarPoints' in row['data']:
                    valid_pts = [p for p in row['data']['lidarPoints'] if isinstance(p, dict) and p.get('isDetected', False)]
                    
                    for pt in valid_pts:
                        if 'position' in pt and 'x' in pt['position'] and 'z' in pt['position']:
                            raw_lidar_x.append(pt['position']['x'])
                            raw_lidar_y.append(pt['position']['z'])
                            
                    # 지면 필터링 적용
                    origin_y = 8.0
                    if 'lidarOrigin' in row['data'] and isinstance(row['data']['lidarOrigin'], dict):
                        origin_y = row['data']['lidarOrigin'].get('y', 8.0)
                        
                    obstacle_pts = filter_ground_points(valid_pts, origin_y)
                    for pt in obstacle_pts:
                        if 'position' in pt and 'x' in pt['position'] and 'z' in pt['position']:
                            lidar_x.append(pt['position']['x'])
                            lidar_y.append(pt['position']['z'])
                            
    fused_x, fused_y = [], []
    if fused_data:
        for row in fused_data:
            if 'data' in row and 'objects' in row['data']:
                for obj in row['data']['objects']:
                    if 'position_map' in obj:
                        pos = obj['position_map']
                        if 'x' in pos and 'y' in pos:
                            fused_x.append(pos['x'])
                            fused_y.append(pos['y'])
                            
    last_obstacles = []
    if obstacles_data:
        for row in reversed(obstacles_data):
            if 'data' in row and 'obstacles' in row['data'] and row['data']['obstacles']:
                last_obstacles = row['data']['obstacles']
                break

    if is_auto:
        # ---------------------------------------------------------
        # Auto Mode Analytics
        # ---------------------------------------------------------
        total_dist = 0.0
        for i in range(1, len(x_vals)):
            total_dist += math.hypot(x_vals[i] - x_vals[i-1], y_vals[i] - y_vals[i-1])
            
        avg_speed = 0.0
        if info_data:
            speeds = [row['data'].get('playerSpeed', 0.0) for row in info_data if 'data' in row and 'playerSpeed' in row['data']]
            if speeds:
                avg_speed = sum(speeds) / len(speeds)
                
        goal_reached = False
        final_dist_to_goal = None
        if dest_data and x_vals and y_vals:
            last_dest = dest_data[-1]
            if 'pose_map' in last_dest and 'pose' in last_dest['pose_map']:
                pos = last_dest['pose_map']['pose']['position']
                gx, gy = pos['x'], pos['y'] 
                last_x, last_y = x_vals[-1], y_vals[-1]
                final_dist_to_goal = math.hypot(gx - last_x, gy - last_y)
                if final_dist_to_goal <= 5.0:
                    goal_reached = True
                    
        collision_count = len(collision_data)
        
        oscillation_count = 0
        if action_data:
            last_cmd_time = None
            last_turn = None
            for row in action_data:
                if 'timestamp_wall' in row and 'response' in row and 'command' in row['response']:
                    ts = row['timestamp_wall']
                    cmd = row['response']['command']
                    current_turn = None
                    if cmd.get('a', False): current_turn = 'A'
                    elif cmd.get('d', False): current_turn = 'D'
                    
                    if current_turn:
                        if last_turn and current_turn != last_turn and last_cmd_time:
                            if (ts - last_cmd_time) < 1.0: 
                                oscillation_count += 1
                        last_turn = current_turn
                        last_cmd_time = ts
                        
        summary_md += "## 1. 자율주행 통합 평가 (Auto Mode Evaluation)\n"
        summary_md += "### 🏆 목표 도달 (Success/Fail)\n"
        if final_dist_to_goal is not None:
            status = "**Success (도달 성공)**" if goal_reached else "**Fail (도달 실패)**"
            summary_md += f"- 상태: {status}\n"
            summary_md += f"- 최종 목표와의 거리: {final_dist_to_goal:.2f}m\n\n"
        else:
            summary_md += "- 목표 지점 설정 데이터(`destination.jsonl`)가 없어 평가할 수 없습니다.\n\n"
            
        summary_md += "### ⏱ 효율성 (Efficiency)\n"
        summary_md += f"- 총 이동 거리: {total_dist:.2f}m\n"
        summary_md += f"- 평균 이동 속도: {avg_speed:.2f}\n\n"
        
        summary_md += "### 🛡 안전성 (Safety)\n"
        summary_md += f"- 총 충돌 횟수: {collision_count}회\n\n"
        
        summary_md += "### ⚖ 안정성 (Stability)\n"
        summary_md += f"- 조향 진동(Oscillation) 발생 횟수: {oscillation_count}회 (1초 이내 좌/우 회전 반복)\n\n"
        
        if x_vals and y_vals:
            fig, ax = plt.subplots(figsize=(10, 8))
            ax.plot(x_vals, y_vals, label='Tank Trajectory', color='blue', linewidth=2)
            if dest_data:
                last_dest = dest_data[-1]
                if 'pose_map' in last_dest and 'pose' in last_dest['pose_map']:
                    pos = last_dest['pose_map']['pose']['position']
                    ax.scatter([pos['x']], [pos['y']], s=200, color='gold', marker='*', label='Goal', zorder=5)
            ax.set_title('Tank Autonomous Trajectory')
            ax.set_xlabel('X')
            ax.set_ylabel('Z (Forward)')
            ax.grid(True)
            ax.legend()
            ax.set_aspect('equal', 'datalim')
            traj_path = report_dir / "trajectory.png"
            plt.savefig(traj_path, dpi=150)
            plt.close()
            summary_md += "## 2. 주행 궤적 (Trajectory)\n"
            summary_md += f"![Trajectory](./trajectory.png)\n\n"

    else:
        # ---------------------------------------------------------
        # Monitor Mode Analytics
        # ---------------------------------------------------------
        if x_vals and y_vals:
            fig, ax = plt.subplots(figsize=(10, 8))
            
            mapped_x, mapped_y = [], []
            unmapped_x, unmapped_y = [], []
            
            pad = 0.5
            for lx, ly in zip(lidar_x, lidar_y):
                hit_installed = False
                for obs in last_obstacles:
                    if (obs['x_min'] - pad <= lx <= obs['x_max'] + pad) and \
                       (obs['z_min'] - pad <= ly <= obs['z_max'] + pad):
                        hit_installed = True
                        break
                
                if hit_installed:
                    mapped_x.append(lx)
                    mapped_y.append(ly)
                else:
                    unmapped_x.append(lx)
                    unmapped_y.append(ly)
            
            if unmapped_x and unmapped_y:
                ax.scatter(unmapped_x, unmapped_y, s=10, color='cyan', alpha=0.8, marker='^', label='Unmapped Hits', zorder=4)
                
            if mapped_x and mapped_y:
                ax.scatter(mapped_x, mapped_y, s=10, color='lime', alpha=1.0, marker='o', label='Mapped Obstacle Hits', zorder=5)
            
            if fused_x and fused_y:
                ax.scatter(fused_x, fused_y, s=80, color='blue', alpha=0.5, marker='*', label='Fused Objects (★)', zorder=6)
            
            ax.plot(x_vals, y_vals, label='Tank Trajectory', color='blue', linewidth=2, zorder=3)
            
            for idx, obs in enumerate(last_obstacles):
                width = obs['x_max'] - obs['x_min']
                height = obs['z_max'] - obs['z_min']
                rect = plt.Rectangle((obs['x_min'], obs['z_min']), width, height, 
                                     fill=True, color='red', alpha=0.5, 
                                     label='Obstacle (Ground Truth)' if idx == 0 else "")
                ax.add_patch(rect)
            
            ax.set_title('Tank Trajectory, LiDAR Hits & Obstacles')
            ax.set_xlabel('X')
            ax.set_ylabel('Z (Forward)')
            ax.grid(True)
            ax.legend()
            ax.set_aspect('equal', 'datalim')
            
            traj_path = report_dir / "trajectory.png"
            plt.savefig(traj_path, dpi=150)
            plt.close()
            
            summary_md += "## 1. 경로 및 장애물 인지 (Trajectory & LiDAR)\n"
            summary_md += f"![Trajectory](./trajectory.png)\n\n"
            summary_md += "전차의 실제 주행 경로(파란 선)와 맵 상의 정답 장애물 구역(빨간 박스), 그리고 라이다 센서가 인지한 장애물 표면입니다.\n"
            summary_md += "라이다 센서가 **맵에 설치된 장애물(빨간 박스 내부)**에 적중한 점은 **형광 초록색 동그라미(Lime)**로 표시했습니다.\n"
            summary_md += "라이다 센서가 **그 외의 위치(빨간 박스 외부)**에 적중한 점은 **형광 하늘색 세모(Cyan)**로 표시했습니다.\n"
            summary_md += "센서 퓨전으로 감지된 최종 좌표는 **파란색 별(★)**로 표시됩니다.\n\n"

            total_obstacles = 0
            detected_count = 0
            missed_count = 0
            out_of_range_count = 0
            occluded_count = 0
            filtered_ground_count = 0
            
            if last_obstacles:
                total_obstacles = len(last_obstacles)
                for obs in last_obstacles:
                    pad = 0.5
                    hits = 0
                    for lx, lz in zip(lidar_x, lidar_y):
                        if (obs['x_min'] - pad <= lx <= obs['x_max'] + pad) and \
                           (obs['z_min'] - pad <= lz <= obs['z_max'] + pad):
                            hits += 1
                            
                    if hits > 0:
                        detected_count += 1
                    else:
                        missed_count += 1
                        raw_hits = 0
                        for rx, rz in zip(raw_lidar_x, raw_lidar_y):
                            if (obs['x_min'] - pad <= rx <= obs['x_max'] + pad) and \
                               (obs['z_min'] - pad <= rz <= obs['z_max'] + pad):
                                raw_hits += 1
                                break
                                
                        if raw_hits > 0:
                            filtered_ground_count += 1
                        else:
                            cx = (obs['x_min'] + obs['x_max']) / 2.0
                            cz = (obs['z_min'] + obs['z_max']) / 2.0
                            min_dist = float('inf')
                            for tx, tz in zip(x_vals, y_vals):
                                d = math.hypot(tx - cx, tz - cz)
                                if d < min_dist:
                                    min_dist = d
                                    
                            if min_dist > 30.0:
                                out_of_range_count += 1
                            else:
                                occluded_count += 1
            
            summary_md += "## 2. 장애물 인지 성능 수치 분석 (Metrics)\n"
            if total_obstacles > 0:
                summary_md += f"- **총 설치된 장애물 수:** {total_obstacles}개\n"
                summary_md += f"- **성공적으로 인지된 장애물 수:** {detected_count}개 (인지율: {detected_count/total_obstacles*100:.1f}%)\n"
                summary_md += f"- **인지 실패(Missed) 장애물 수:** {missed_count}개\n\n"
                if missed_count > 0:
                    summary_md += "### 미인지 원인 요약\n"
                    summary_md += f"- **지면으로 오인되어 필터링됨 (납작한 장애물 / 20cm 이하):** {filtered_ground_count}개\n"
                    summary_md += f"- **탐지 거리 초과 (라이다 사거리 30m 밖):** {out_of_range_count}개\n"
                    summary_md += f"- **사각지대 또는 가려짐 (사거리 내에 있지만 다른 물체에 막힘):** {occluded_count}개\n\n"

                unmapped_hits_count = len(unmapped_x)
                mapped_hits_count = len(mapped_x)
                total_pts = unmapped_hits_count + mapped_hits_count
                
                if total_pts > 0:
                    summary_md += "### 정답 구역 내/외부 인지율 분석\n"
                    summary_md += "라이다가 지면을 제외하고 인식한 전체 입체물 점 중에서, 정답 구역(빨간 박스)과 그 외 구역의 비율입니다.\n\n"
                    summary_md += f"- **전체 라이다 인지 점 개수:** {total_pts}개\n"
                    summary_md += f"- **정답 장애물(빨간 박스 내부) 적중 점:** {mapped_hits_count}개 ({mapped_hits_count/total_pts*100:.1f}%) - 형광 초록색(Lime)\n"
                    summary_md += f"- **그 외 구역(빨간 박스 외부) 적중 점:** {unmapped_hits_count}개 ({unmapped_hits_count/total_pts*100:.1f}%) - 형광 하늘색(Cyan)\n\n"
            else:
                summary_md += "맵에 설치된 장애물이 없습니다.\n\n"
                
            summary_md += "## 3. 센서 퓨전 (Sensor Fusion) 정확도 분석\n"
            if not fused_data:
                summary_md += "fused.jsonl 로그가 없어 센서 퓨전 분석을 건너뜁니다.\n\n"
            else:
                tp_obstacles = 0
                fn_obstacles = 0
                fp_points = 0
                total_fused_points = len(fused_x)
                
                avg_errors = []
                
                if last_obstacles:
                    for obs in last_obstacles:
                        cx = (obs['x_min'] + obs['x_max']) / 2.0
                        cz = (obs['z_min'] + obs['z_max']) / 2.0
                        
                        dists = []
                        for fx, fy in zip(fused_x, fused_y):
                            d = math.hypot(fx - cx, fy - cz)
                            if d <= 1.5:
                                dists.append(d)
                        
                        if dists:
                            tp_obstacles += 1
                            avg_errors.append(sum(dists) / len(dists))
                        else:
                            fn_obstacles += 1
                            
                    for fx, fy in zip(fused_x, fused_y):
                        is_fp = True
                        for obs in last_obstacles:
                            cx = (obs['x_min'] + obs['x_max']) / 2.0
                            cz = (obs['z_min'] + obs['z_max']) / 2.0
                            if math.hypot(fx - cx, fy - cz) <= 1.5:
                                is_fp = False
                                break
                        if is_fp:
                            fp_points += 1
                            
                    summary_md += f"- **성공적으로 퓨전된(True Positive) 장애물:** {tp_obstacles}개 / 전체 {total_obstacles}개\n"
                    if avg_errors:
                        summary_md += f"- **퓨전 중심점 평균 거리 오차:** {sum(avg_errors)/len(avg_errors):.2f}m\n"
                    summary_md += f"- **미탐지(False Negative) 장애물:** {fn_obstacles}개\n"
                    summary_md += f"- **오인식/고스트(False Positive) 퓨전 점 수:** {fp_points}개 (전체 퓨전 출력 {total_fused_points}번 중)\n\n"

    # 제어 명령 분포 (공통)
    if action_data:
        w_counts = s_counts = a_counts = d_counts = 0
        for row in action_data:
            if 'response' in row and 'command' in row['response']:
                cmd = row['response']['command']
                if cmd.get('w', False): w_counts += 1
                if cmd.get('s', False): s_counts += 1
                if cmd.get('a', False): a_counts += 1
                if cmd.get('d', False): d_counts += 1
                
        commands = ['W (Forward)', 'S (Backward)', 'A (Left)', 'D (Right)']
        counts = [w_counts, s_counts, a_counts, d_counts]
        
        plt.figure(figsize=(8,6))
        plt.bar(commands, counts, color=['green', 'red', 'blue', 'orange'])
        plt.title('Control Command Distribution')
        plt.ylabel('Command Count')
        
        cmd_path = report_dir / "commands.png"
        plt.savefig(cmd_path)
        plt.close()
        
        summary_md += "## 제어 명령(Control Linearity) 빈도 분석\n"
        summary_md += f"![Commands](./commands.png)\n\n"
        summary_md += "W/S/A/D 키 명령이 시뮬레이터로 전달된 횟수입니다. 직진성, 조향 빈도를 파악할 수 있습니다.\n\n"
        
    summary_path = report_dir / "summary.md"
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write(summary_md)
        
    print(f"분석 완료. 결과가 {report_dir} 에 저장되었습니다.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Analyze Tank Logs')
    parser.add_argument('--map', type=str, default='알 수 없음 (통신 데이터 누락)', help='사용된 맵 이름')
    parser.add_argument('--target_mode', type=str, default='latest', choices=['latest', 'monitor', 'auto'], help='분석 대상 세션 모드')
    args = parser.parse_args()
    
    analyze_and_plot(map_name=args.map, target_mode=args.target_mode)
