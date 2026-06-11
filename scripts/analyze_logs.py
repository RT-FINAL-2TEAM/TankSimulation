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
    fused_file = latest_session / "fused.jsonl"
    
    info_data = parse_jsonl(info_file)
    action_data = parse_jsonl(action_file)
    obstacles_data = parse_jsonl(obstacles_file)
    fused_data = parse_jsonl(fused_file)
    
    summary_md = f"# 주행 분석 결과 보고서\n\n- 생성일시: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
    summary_md += f"- 분석 대상 로그 세션: `{latest_session.name}`\n"
    summary_md += f"- 사용된 맵: **{map_name}**\n"
    summary_md += f"- info 로그 개수: {len(info_data)}\n"
    summary_md += f"- action 로그 개수: {len(action_data)}\n\n"
    
    # 1. Trajectory (경로), 장애물(Obstacles), LiDAR 인지 점 시각화
    if info_data:
        x_vals, y_vals = [], []
        lidar_x, lidar_y = [], []
        raw_lidar_x, raw_lidar_y = [], []
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
                    
                    for pt in valid_pts:
                        if 'position' in pt and 'x' in pt['position'] and 'z' in pt['position']:
                            raw_lidar_x.append(pt['position']['x'])
                            raw_lidar_y.append(pt['position']['z'])
                            
                    # 지면 필터링 적용 (실제 장애물에 부딪힌 점만 추출)
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
        
        if x_vals and y_vals:
            fig, ax = plt.subplots(figsize=(10, 8))
            
            # 마지막 장애물 목록 미리 찾기
            last_obstacles = []
            if obstacles_data:
                for row in reversed(obstacles_data):
                    if 'data' in row and 'obstacles' in row['data'] and row['data']['obstacles']:
                        last_obstacles = row['data']['obstacles']
                        break
            
            # 지면이 필터링된 라이다 점들을 정답 구역(빨간 박스) 내부/외부로 분리
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
            
            # 빨간 박스 외부 적중 포인트 (다른 형광색 - Cyan)
            if unmapped_x and unmapped_y:
                ax.scatter(unmapped_x, unmapped_y, s=10, color='cyan', alpha=0.8, marker='^', label='Unmapped Hits', zorder=4)
                
            # 빨간 박스 내부 적중 포인트 (초록 형광색 - Lime)
            if mapped_x and mapped_y:
                ax.scatter(mapped_x, mapped_y, s=10, color='lime', alpha=1.0, marker='o', label='Mapped Obstacle Hits', zorder=5)
            
            # 센서 퓨전 결과 플로팅
            if fused_x and fused_y:
                ax.scatter(fused_x, fused_y, s=80, color='blue', alpha=0.5, marker='*', label='Fused Objects (★)', zorder=6)
            
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
            summary_md += "전차의 실제 주행 경로(파란 선)와 맵 상의 정답 장애물 구역(빨간 박스), 그리고 라이다 센서가 인지한 장애물 표면입니다.\n"
            summary_md += "라이다 센서가 **맵에 설치된 장애물(빨간 박스 내부)**에 적중한 점은 **형광 초록색 동그라미(Lime)**로 표시했습니다.\n"
            summary_md += "라이다 센서가 **그 외의 위치(빨간 박스 외부)**에 적중한 점은 **형광 하늘색 세모(Cyan)**로 표시했습니다.\n"
            summary_md += "센서 퓨전으로 감지된 최종 좌표는 **파란색 별(★)**로 표시됩니다.\n\n"
            # ---------------------------------------------------------
            # 1.5 수치적 장애물 인지 분석 (Detection Metrics)
            # ---------------------------------------------------------
            total_obstacles = 0
            detected_count = 0
            missed_count = 0
            out_of_range_count = 0
            occluded_count = 0
            filtered_ground_count = 0
            
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
                                # 미인지 원인 분석: 원래 라이다에 찍혔으나 지면으로 필터링되었는지 확인
                                raw_hits = 0
                                for rx, rz in zip(raw_lidar_x, raw_lidar_y):
                                    if (obs['x_min'] - pad <= rx <= obs['x_max'] + pad) and \
                                       (obs['z_min'] - pad <= rz <= obs['z_max'] + pad):
                                        raw_hits += 1
                                        break
                                        
                                if raw_hits > 0:
                                    filtered_ground_count += 1
                                else:
                                    # 경로와의 최소 거리 계산
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
                        break
            
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
                # ---------------------------------------------------------
                # 1.6 구역 내/외부 적중률 분석
                # ---------------------------------------------------------
                unmapped_hits_count = len(unmapped_x) if 'unmapped_x' in locals() else 0
                mapped_hits_count = len(mapped_x) if 'mapped_x' in locals() else 0
                total_pts = unmapped_hits_count + mapped_hits_count
                
                if total_pts > 0:
                    summary_md += "### 정답 구역 내/외부 인지율 분석\n"
                    summary_md += "라이다가 지면을 제외하고 인식한 전체 입체물 점 중에서, 정답 구역(빨간 박스)과 그 외 구역의 비율입니다.\n\n"
                    summary_md += f"- **전체 라이다 인지 점 개수:** {total_pts}개\n"
                    summary_md += f"- **정답 장애물(빨간 박스 내부) 적중 점:** {mapped_hits_count}개 ({mapped_hits_count/total_pts*100:.1f}%) - 형광 초록색(Lime)\n"
                    summary_md += f"- **그 외 구역(빨간 박스 외부) 적중 점:** {unmapped_hits_count}개 ({unmapped_hits_count/total_pts*100:.1f}%) - 형광 하늘색(Cyan)\n\n"
            else:
                summary_md += "맵에 설치된 장애물이 없습니다.\n\n"
                
            # ---------------------------------------------------------
            # 센서 퓨전 (Sensor Fusion) 정확도 분석
            # ---------------------------------------------------------
            summary_md += "## 3. 센서 퓨전 (Sensor Fusion) 정확도 분석\n"
            if not fused_data:
                summary_md += "fused.jsonl 로그가 없어 센서 퓨전 분석을 건너뜁니다.\n\n"
            else:
                tp_obstacles = 0
                fn_obstacles = 0
                fp_points = 0
                total_fused_points = len(fused_x)
                
                avg_errors = []
                
                # 1. TP / FN 계산 (GT 기준)
                if 'last_obstacles' in locals() and last_obstacles:
                    for obs in last_obstacles:
                        cx = (obs['x_min'] + obs['x_max']) / 2.0
                        cz = (obs['z_min'] + obs['z_max']) / 2.0
                        
                        # 이 장애물에 대해 1.5m 이내의 퓨전 점 찾기
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
                            
                    # 2. FP 계산 (Fused 점 기준)
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
                
            # ---------------------------------------------------------
            
    # 4. Control (제어) 명령 분포
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
