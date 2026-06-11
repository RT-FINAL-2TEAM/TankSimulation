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

from lidar.config import TOPIC_LIDAR_POINTS

# 프로젝트 구조에 맞춘 토픽 설정
TOPIC_FUSED_OBJECTS = "/tank/perception/fused_objects"
TOPIC_PLAYER_POSE = "/tank/player/pose"

class PolarViewer(Node):
    def __init__(self):
        super().__init__("polar_scan_viewer")
        
        # 1. LiDAR 포인트 구독
        self.subscription_lidar = self.create_subscription(
            String, TOPIC_LIDAR_POINTS, self.callback_lidar, 10
        )
        
        # 2. 융합된 객체(YOLO + LiDAR Cluster) 구독
        self.subscription_objects = self.create_subscription(
            String, TOPIC_FUSED_OBJECTS, self.callback_fused, 10
        )

        # 3. 객체의 상대 각도 계산을 위한 전차 위치 구독
        self.subscription_pose = self.create_subscription(
            PoseStamped, TOPIC_PLAYER_POSE, self.callback_pose, 10
        )
        
        self.lock = threading.Lock()

        # 전차 위치 캐싱용 변수
        self.tank_x = 0.0
        self.tank_y = 0.0

        # 데이터 저장용
        self.channels = {ch: {"theta": [], "radius": []} for ch in range(1, 9)}
        self.visible = {ch: True for ch in range(1, 9)}
        self.detected_objects = []
        
        self.colors = {
            1: "red", 2: "blue", 3: "green", 4: "orange",
            5: "purple", 6: "brown", 7: "pink", 8: "black",
        }

        # UI 초기화
        plt.ion()
        self.fig = plt.figure(figsize=(11, 8))
        self.ax = self.fig.add_axes([0.05, 0.05, 0.72, 0.90], projection="polar")
        check_ax = self.fig.add_axes([0.82, 0.20, 0.15, 0.60])

        labels = [f"CH{i}" for i in range(1, 9)]
        states = [True] * 8
        self.check = CheckButtons(check_ax, labels, states)
        self.check.on_clicked(self.on_click)
        self.fig.canvas.mpl_connect("close_event", self.on_close)
        
        self.running = True

        # Scatter 객체 초기화 (렌더링 최적화)
        self.scatters = {}
        for ch in range(1, 9):
            self.scatters[ch] = self.ax.scatter([], [], s=5, color=self.colors[ch], label=f"CH{ch}")
            
        # 융합된 객체용 마커 (별모양)
        self.obj_scatter = self.ax.scatter([], [], s=200, marker='*', color='magenta', edgecolors='black', zorder=5, label='Detected Object')
        self.obj_texts = []
        
        # Polar 좌표계 방향 설정 (N이 0도, 시계방향)
        self.ax.set_theta_zero_location("N")
        self.ax.set_theta_direction(-1)
        self.ax.grid(True)
        self.ax.legend(loc="upper right", fontsize=8)
        self.title_text = self.ax.set_title("LiDAR Polar Scan (0 points)")

    def on_close(self, event):
        self.running = False

    def on_click(self, label):
        ch = int(label.replace("CH", ""))
        self.visible[ch] = not self.visible[ch]

    # --- 콜백 함수들 ---
    def callback_pose(self, msg):
        # 전차의 맵 상 좌표 업데이트
        self.tank_x = msg.pose.position.x
        self.tank_y = msg.pose.position.y

    def callback_lidar(self, msg):
        try:
            payload = json.loads(msg.data)
        except Exception:
            return

        points = payload.get("points", [])
        temp = {ch: {"theta": [], "radius": []} for ch in range(1, 9)}

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
            
    def callback_fused(self, msg):
        try:
            payload = json.loads(msg.data)
            objects = payload.get("objects", [])
            parsed_objects = []
            
            for obj in objects:
                pos = obj.get("position_map", {})
                obj_x = float(pos.get("x", 0.0))
                obj_y = float(pos.get("y", 0.0))
                class_name = obj.get("className", "Unknown")
                heading = float(obj.get("camera_heading_deg", 0.0))
                
                dx = obj_x - self.tank_x
                dy = obj_y - self.tank_y
                
                # [수정] Z축 높이를 제외한 2D 평면 상의 실제 거리를 계산합니다.
                distance_2d = math.hypot(dx, dy)
                
                global_bearing = math.degrees(math.atan2(dx, dy))
                rel_bearing_deg = global_bearing - heading
                rel_bearing_deg = (rel_bearing_deg + 180.0) % 360.0 - 180.0
                
                parsed_objects.append({
                    "theta": math.radians(rel_bearing_deg),
                    "radius": distance_2d,  # 3D distance 대신 2D distance 사용
                    "class_name": class_name
                })
                    
            with self.lock:
                self.detected_objects = parsed_objects
                
        except Exception as e:
            self.get_logger().debug(f"Fused objects parse failed: {e}")
            
    # --- 렌더링 ---
    def update(self):
        with self.lock:
            total_points = 0
            max_r = 1.0

            # 1. LiDAR 포인트 업데이트
            for ch in range(1, 9):
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

            # 2. 객체 마커 및 텍스트 업데이트
            for txt in self.obj_texts:
                txt.remove()
            self.obj_texts.clear()
            
            if self.detected_objects:
                obj_thetas = []
                obj_radii = []
                
                for obj in self.detected_objects:
                    t = obj["theta"]
                    r = obj["radius"]
                    obj_thetas.append(t)
                    obj_radii.append(r)
                    
                    if r > max_r:
                        max_r = r
                        
                    # 텍스트 라벨 추가 (클래스명 표시)
                    txt = self.ax.text(t, r + 2.0, obj["class_name"], 
                                       fontsize=11, fontweight='bold', color='magenta', 
                                       ha='center', va='bottom')
                    self.obj_texts.append(txt)
                
                self.obj_scatter.set_offsets(np.column_stack([obj_thetas, obj_radii]))
            else:
                self.obj_scatter.set_offsets(np.empty((0, 2)))

            self.ax.set_rmax(max_r + 5.0)
            self.title_text.set_text(f"LiDAR Polar Scan ({total_points} points) | Detected Objects: {len(self.detected_objects)}")

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