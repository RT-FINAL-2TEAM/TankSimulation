import os
import json
import glob
import sys
import math
import argparse
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt

# 로컬 패키지 import를 위해 src 경로 추가
sys.path.append("c:/dev/rotem/tank_project/src/lidar")
from lidar.perception_utils import filter_ground_points

def create_report_dir(session_name):
    base_dir = Path("c:/dev/rotem/tank_project/reports")
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

def analyze_and_plot(logs_dir="c:/dev/rotem/tank_project/tank_logs", map_name="알 수 없음 (시뮬레이터 제공 안됨)"):
    logs_path = Path(logs_dir)
    if not logs_path.exists():
        print(f"로그 폴더를 찾을 수 없습니다: {logs_dir}")
        return
        
    session_dirs = [d for d in logs_path.iterdir() if d.is_dir() and d.name.startswith("session_")]
    if not session_dirs:
        print(f"분석할 세션 폴더(session_*)가 {logs_dir} 에 없습니다.")
        return
        
    latest_session = max(session_dirs, key=lambda x: x.stat().st_mtime)
    print(f"가장 최근 세션 분석 시작: {latest_session.name}")
    
    report_dir = create_report_dir(latest_session.name)
    
    info_file = latest_session / "info.jsonl"
    action_file = latest_session / "get_action.jsonl"
    obstacles_file = latest_session / "obstacles.jsonl"
    
    info_data = parse_jsonl(info_file)
    action_data = parse_jsonl(action_file)
    obstacles_data = parse_jsonl(obstacles_file)
    
    summary_md = f"# 주행 분석 결과 보고서\n\n- 생성일시: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
    summary_md += f"- 분석 대상 로그 세션: `{latest_session.name}`\n"
    summary_md += f"- 사용된 맵: **{map_name}**\n"
    summary_md += f"- info 로그 개수: {len(info_data)}\n"
    summary_md += f"- action 로그 개수: {len(action_data)}\n\n"
    
    # 1. Trajectory (경로), 장애물(Obstacles), LiDAR 인지 점 시각화
    if info_data:
        x_vals, y_vals = [], []
        lidar_x, lidar_y = [], []
        for row in info_data:
            if 'data' in row and isinstance(row['data'], dict):
                # 1) 경로 추출
                if 'playerPos' in row['data']:
                    pos = row['data']['playerPos']
                    if 'x' in pos and 'z' in pos:
                        x_vals.append(pos['x'])
                        y_vals.append(pos['z'])
                
                # 2) LiDAR 포인트 추출 (장애물에 맞은 점들)
                if 'lidarPoints' in row['data']:
                    valid_pts = [p for p in row['data']['lidarPoints'] if isinstance(p, dict) and p.get('isDetected', False)]
                    
                    # 지면 필터링 적용 (실제 장애물에 부딪힌 점만 추출)
                    origin_y = 8.0
                    if 'lidarOrigin' in row['data'] and isinstance(row['data']['lidarOrigin'], dict):
                        origin_y = row['data']['lidarOrigin'].get('y', 8.0)
                        
                    obstacle_pts = filter_ground_points(valid_pts, origin_y)
                    
                    for pt in obstacle_pts:
                        if 'position' in pt and 'x' in pt['position'] and 'z' in pt['position']:
                            lidar_x.append(pt['position']['x'])
                            lidar_y.append(pt['position']['z'])
        
        if x_vals and y_vals:
            fig, ax = plt.subplots(figsize=(10, 8))
            
            # 마지막 장애물 목록 미리 찾기
            last_obstacles = []
            if obstacles_data:
                for row in reversed(obstacles_data):
                    if 'data' in row and 'obstacles' in row['data'] and row['data']['obstacles']:
                        last_obstacles = row['data']['obstacles']
                        break
            
            # 라이다 점들을 지형지물 vs 설치된 장애물로 분리
            installed_x, installed_y = [], []
            terrain_x, terrain_y = [], []
            
            pad = 0.5
            for lx, ly in zip(lidar_x, lidar_y):
                hit_installed = False
                for obs in last_obstacles:
                    if (obs['x_min'] - pad <= lx <= obs['x_max'] + pad) and \
                       (obs['z_min'] - pad <= ly <= obs['z_max'] + pad):
                        hit_installed = True
                        break
                
                if hit_installed:
                    installed_x.append(lx)
                    installed_y.append(ly)
                else:
                    terrain_x.append(lx)
                    terrain_y.append(ly)
            
            # 지형지물 포인트 플로팅
            if terrain_x and terrain_y:
                ax.scatter(terrain_x, terrain_y, s=10, color='cyan', alpha=0.8, marker='^', label='Terrain Hits', zorder=4)
                
            # 설치된 장애물 포인트 플로팅
            if installed_x and installed_y:
                ax.scatter(installed_x, installed_y, s=10, color='lime', alpha=1.0, marker='o', label='Installed Obstacles', zorder=5)
            
            # 전차 궤적
            ax.plot(x_vals, y_vals, label='Tank Trajectory', color='blue', linewidth=2, zorder=3)
            
            # 장애물 그리기 (가장 마지막에 업데이트된 장애물 목록 기준)
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
            
            # 비율 동일하게 맞추기 (맵 왜곡 방지)
            ax.set_aspect('equal', 'datalim')
            
            traj_path = report_dir / "trajectory.png"
            plt.savefig(traj_path, dpi=150)
            plt.close()
            
            summary_md += "## 1. 경로 및 장애물 인지 (Trajectory & LiDAR)\n"
            summary_md += f"![Trajectory](./trajectory.png)\n\n"
            summary_md += "전차의 실제 주행 경로(파란 선)와 맵에 설치된 실제 장애물(빨간 박스), 그리고 라이다 센서가 인식한 표면입니다.\n"
            summary_md += "설치된 장애물 표면에 적중한 점은 **형광 초록색 동그라미(Lime)**로, 자연 지형지물(언덕/나무 등)에 적중한 점은 **하늘색 세모(Cyan)**로 구분하여 표시했습니다.\n\n"
            # ---------------------------------------------------------
            # 1.5 수치적 장애물 인지 분석 (Detection Metrics)
            # ---------------------------------------------------------
            total_obstacles = 0
            detected_count = 0
            missed_count = 0
            missed_details = []
            
            if obstacles_data:
                for row in reversed(obstacles_data):
                    if 'data' in row and 'obstacles' in row['data'] and row['data']['obstacles']:
                        last_obstacles = row['data']['obstacles']
                        total_obstacles = len(last_obstacles)
                        
                        for obs in last_obstacles:
                            # 해당 장애물 내부에 들어온 라이다 점 개수 확인 (약간의 여유 마진 0.5m)
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
                                # 미인지 원인 분석 (경로와의 최소 거리 계산)
                                cx = (obs['x_min'] + obs['x_max']) / 2.0
                                cz = (obs['z_min'] + obs['z_max']) / 2.0
                                min_dist = float('inf')
                                for tx, tz in zip(x_vals, y_vals):
                                    d = math.hypot(tx - cx, tz - cz)
                                    if d < min_dist:
                                        min_dist = d
                                        
                                if min_dist > 14.5:
                                    reason = f"탐지 거리 초과 (경로와의 최소 거리 {min_dist:.1f}m > 라이다 사거리 15m)"
                                else:
                                    reason = f"사각지대 또는 가려짐 (경로와 {min_dist:.1f}m로 가깝지만 빔이 닿지 않음)"
                                missed_details.append(f"- 장애물 중심({cx:.1f}, {cz:.1f}): {reason}")
                        break
            
            summary_md += "## 2. 장애물 인지 성능 수치 분석 (Metrics)\n"
            if total_obstacles > 0:
                summary_md += f"- **총 설치된 장애물 수:** {total_obstacles}개\n"
                summary_md += f"- **성공적으로 인지된 장애물 수:** {detected_count}개 (인지율: {detected_count/total_obstacles*100:.1f}%)\n"
                summary_md += f"- **인지 실패(Missed) 장애물 수:** {missed_count}개\n\n"
                if missed_count > 0:
                    summary_md += "### 미인지 원인 분석\n"
                    for detail in missed_details:
                        summary_md += f"{detail}\n"
                    summary_md += "\n"
                # ---------------------------------------------------------
                # 1.6 지형지물 vs 설치 장애물 구분 분석 (Terrain vs Dynamic Obstacles)
                # ---------------------------------------------------------
                terrain_hits_count = 0
                installed_hits_count = 0
                
                # 모든 라이다 점에 대해 빨간 박스 내부에 있는지 검사
                # 앞서 last_obstacles가 추출되었을 때만 수행 가능
                if 'last_obstacles' in locals() and last_obstacles:
                    pad = 0.5
                    for lx, lz in zip(lidar_x, lidar_y):
                        hit_installed = False
                        for obs in last_obstacles:
                            if (obs['x_min'] - pad <= lx <= obs['x_max'] + pad) and \
                               (obs['z_min'] - pad <= lz <= obs['z_max'] + pad):
                                hit_installed = True
                                break
                        
                        if hit_installed:
                            installed_hits_count += 1
                        else:
                            terrain_hits_count += 1
                
                total_pts = len(lidar_x)
                if total_pts > 0 and 'last_obstacles' in locals() and last_obstacles:
                    summary_md += "### 지형지물 vs 설치 장애물 구분 분석 (Terrain vs Dynamic Obstacles)\n"
                    summary_md += "험지(울퉁불퉁한 지형) 테스트의 경우, 라이다가 인식한 전체 장애물 점 중에서 사용자가 설치한 장애물 외의 자연 지형지물(언덕, 나무, 바위 등)을 얼마나 인식했는지 수치화합니다.\n\n"
                    summary_md += f"- **전체 라이다 장애물 인지 점 개수:** {total_pts}개\n"
                    summary_md += f"- **설치된 장애물(빨간 박스)에 적중한 점:** {installed_hits_count}개 ({installed_hits_count/total_pts*100:.1f}%) - 형광 초록색(Lime)\n"
                    summary_md += f"- **자연 지형지물(험지/언덕 등)로 인식된 점:** {terrain_hits_count}개 ({terrain_hits_count/total_pts*100:.1f}%) - 하늘색 세모(Cyan)\n\n"
                    summary_md += "> 💡 **분석 포인트:** 위 그래프에서 빨간 박스가 없는 곳에 찍힌 하늘색 세모 점들이 바로 '자연 지형지물'을 장애물로 정상 인식한 결과입니다. 지형지물 점의 비율을 통해 현재 맵의 험지(난이도) 수준 및 라이다의 지형 인지 능력을 확인할 수 있습니다.\n\n"
            else:
                summary_md += "맵에 설치된 장애물이 없습니다.\n\n"
                
            # ---------------------------------------------------------
            
    # 3. Control (제어) 명령 분포
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
        summary_md += "## 2. 제어 명령(Control Linearity) 빈도 분석\n"
        summary_md += f"![Commands](./commands.png)\n\n"
        summary_md += "W/S/A/D 키 명령이 시뮬레이터로 전달된 횟수입니다. 직진성, 조향 빈도를 파악할 수 있습니다.\n\n"
        
    summary_path = report_dir / "summary.md"
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write(summary_md)
        
    print(f"분석 완료. 결과가 {report_dir} 에 저장되었습니다.")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Analyze Tank Logs')
    parser.add_argument('--map', type=str, default='알 수 없음 (통신 데이터 누락)', help='사용된 맵 이름')
    args = parser.parse_args()
    
    analyze_and_plot(map_name=args.map)
