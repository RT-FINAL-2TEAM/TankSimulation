#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import math
import threading
import time

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import CheckButtons

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import PoseStamped

# 사용 토픽 설정
TOPIC_LIDAR_POINTS = "/tank/sensor/lidar/points"
TOPIC_PLAYER_POSE = "/tank/player/pose"
TOPIC_PLAYER_STATE = "/tank/player/state"
TOPIC_CLUSTERS = "/tank/visual_perception/lidar_clusters"

class PolarViewer(Node):
    def __init__(self):
        super().__init__("polar_scan_viewer")
        
        # 1. LiDAR 포인트 구독
        self.subscription_lidar = self.create_subscription(
            String, TOPIC_LIDAR_POINTS, self.callback_lidar, 10
        )
        
        # 2. 전차 위치 구독 (클러스터 상대 좌표 계산용)
        self.subscription_pose = self.create_subscription(
            PoseStamped, TOPIC_PLAYER_POSE, self.callback_pose, 10
        )
        
        # 3. 전차 상태 구독 (전차의 Heading 각도 계산용)
        self.subscription_state = self.create_subscription(
            String, TOPIC_PLAYER_STATE, self.callback_state, 10
        )
        
        # 4. LiDAR 기반 클러스터링 구독
        self.subscription_clusters = self.create_subscription(
            String, TOPIC_CLUSTERS, self.callback_clusters, 10
        )
        
        self.lock = threading.Lock()

        # 전차 상태 캐싱용 변수
        self.tank_x = 0.0
        self.tank_y = 0.0
        self.tank_heading = 0.0

        # 데이터 저장용
        self.channels = {ch: {"theta": [], "radius": []} for ch in range(1, 17)}
        self.visible = {ch: True for ch in range(1, 17)}
        self.clusters = []
        
        # 16개 채널을 위한 색상 지정
        self.colors = {
            1: "red", 2: "blue", 3: "green", 4: "orange",
            5: "purple", 6: "brown", 7: "pink", 8: "black",
            9: "cyan", 10: "magenta", 11: "yellow", 12: "lime",
            13: "teal", 14: "navy", 15: "maroon", 16: "olive"
        }

        # UI 초기화
        plt.ion()
        self.fig = plt.figure(figsize=(12, 8))
        self.ax = self.fig.add_axes([0.05, 0.05, 0.70, 0.90], projection="polar")
        check_ax = self.fig.add_axes([0.80, 0.10, 0.15, 0.80])

        labels = [f"CH{i}" for i in range(1, 17)]
        states = [True] * 16
        self.check = CheckButtons(check_ax, labels, states)
        self.check.on_clicked(self.on_click)
        self.fig.canvas.mpl_connect("close_event", self.on_close)
        
        self.running = True

        # Scatter 객체 초기화 (렌더링 최적화)
        self.scatters = {}
        for ch in range(1, 17):
            self.scatters[ch] = self.ax.scatter([], [], s=5, color=self.colors[ch], label=f"CH{ch}")
            
        # 클러스터 마커 초기화 (비어있는 원형 마커)
        self.cluster_scatter = self.ax.scatter([], [], s=200, marker='o', facecolors='none', edgecolors='magenta', linewidth=2, zorder=5, label='Cluster')
        self.cluster_texts = []
            
        # Polar 좌표계 방향 설정 (N이 0도, 시계방향)
        self.ax.set_theta_zero_location("N")
        self.ax.set_theta_direction(-1)
        self.ax.grid(True)
        self.ax.legend(loc="upper right", bbox_to_anchor=(1.15, 1.1), fontsize=8)
        self.title_text = self.ax.set_title("LiDAR 16CH Scan & Clusters")

    def on_close(self, event):
        self.running = False

    def on_click(self, label):
        ch = int(label.replace("CH", ""))
        self.visible[ch] = not self.visible[ch]

    # --- 콜백 함수들 ---
    def callback_pose(self, msg):
        self.tank_x = msg.pose.position.x
        self.tank_y = msg.pose.position.y

    def callback_state(self, msg):
        try:
            data = json.loads(msg.data)
            body = data.get("body", {})
            if "x" in body:
                self.tank_heading = float(body["x"])
            elif "playerBodyX" in data:
                self.tank_heading = float(data["playerBodyX"])
        except Exception:
            pass

    def callback_lidar(self, msg):
        try:
            payload = json.loads(msg.data)
        except Exception:
            return

        points = payload.get("points", [])
        temp = {ch: {"theta": [], "radius": []} for ch in range(1, 17)}

        for point in points:
            if not isinstance(point, dict) or not point.get("isDetected", False):
                continue
            try:
                ch = int(point.get("channelIndex", 1))
            except Exception:
                continue

            if ch not in temp:
                continue

            angle_deg = float(point.get("angle", 0.0))
            distance = float(point.get("distance", 0.0))

            temp[ch]["theta"].append(math.radians(angle_deg))
            temp[ch]["radius"].append(distance)

        with self.lock:
            self.channels = temp

    def callback_clusters(self, msg):
        try:
            payload = json.loads(msg.data)
            clusters_data = payload.get("clusters", [])
            parsed_clusters = []
            
            for c in clusters_data:
                c_id = c.get("id", 0)
                count = c.get("count", 0)
                centroid = c.get("centroid", {})
                
                cx = float(centroid.get("x", 0.0))
                cy = float(centroid.get("y", 0.0))
                
                # 맵 절대 좌표를 전차 기준 상대 좌표로 변환
                dx = cx - self.tank_x
                dy = cy - self.tank_y
                distance_2d = math.hypot(dx, dy)
                
                # y축이 앞, x축이 오른쪽일 때의 방위각 계산 (북쪽 0도 기준)
                global_bearing = math.degrees(math.atan2(dx, dy))
                rel_bearing_deg = global_bearing - self.tank_heading
                
                # -180 ~ 180도로 정규화
                rel_bearing_deg = (rel_bearing_deg + 180.0) % 360.0 - 180.0
                
                parsed_clusters.append({
                    "id": c_id,
                    "count": count,
                    "theta": math.radians(rel_bearing_deg),
                    "radius": distance_2d
                })
                
            with self.lock:
                self.clusters = parsed_clusters
                
        except Exception as e:
            self.get_logger().debug(f"Clusters parse failed: {e}")

    # --- 렌더링 ---
    def update(self):
        with self.lock:
            total_points = 0
            max_r = 1.0

            # 1. LiDAR 포인트 업데이트
            for ch in range(1, 17):
                if not self.visible[ch] or not self.channels[ch]["theta"]:
                    self.scatters[ch].set_offsets(np.empty((0, 2)))
                    continue

                theta = self.channels[ch]["theta"]
                radius = self.channels[ch]["radius"]
                total_points += len(theta)

                offsets = np.column_stack([theta, radius])
                self.scatters[ch].set_offsets(offsets)

                r = max(radius)
                if r > max_r:
                    max_r = r

            # 2. 클러스터 텍스트 및 마커 업데이트
            for txt in self.cluster_texts:
                txt.remove()
            self.cluster_texts.clear()
            
            if self.clusters:
                c_thetas = []
                c_radii = []
                
                for c in self.clusters:
                    t = c["theta"]
                    r = c["radius"]
                    c_thetas.append(t)
                    c_radii.append(r)
                    
                    if r > max_r:
                        max_r = r
                        
                    # RViz와 유사하게 'C번호 N=개수' 텍스트 표시
                    txt = self.ax.text(t, r + 1.5, f"C{c['id']} N={c['count']}", 
                                       fontsize=10, fontweight='bold', color='magenta', 
                                       ha='center', va='bottom')
                    self.cluster_texts.append(txt)
                    
                self.cluster_scatter.set_offsets(np.column_stack([c_thetas, c_radii]))
            else:
                self.cluster_scatter.set_offsets(np.empty((0, 2)))

            self.ax.set_rmax(max_r + 5.0)
            self.title_text.set_text(f"LiDAR 16CH Scan ({total_points} pts) | Clusters: {len(self.clusters)}")

        plt.draw()
        plt.pause(0.001)

def main(args=None):
    rclpy.init(args=args)
    node = PolarViewer()
    
    spin_thread = threading.Thread(
        target=rclpy.spin,
        args=(node,),
        daemon=True,
    )
    spin_thread.start()
    
    try:
        while rclpy.ok() and node.running:
            node.update()
            time.sleep(0.03)
    except KeyboardInterrupt:
        pass
        
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()