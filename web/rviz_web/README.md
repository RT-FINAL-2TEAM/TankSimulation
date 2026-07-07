# rviz_web

Standalone RViz-like 3D web viewer for the Tank Challenge ROS2 workspace.

Coordinate policy:
- Unity raw: x=left/right, y=height, z=forward/back
- ROS tank_map: x=raw.x, y=raw.z, z=raw.y
- Three.js: x=map.x, y=map.z, z=-map.y

Run:
```bash
ros2 launch rviz_web rviz_web.launch.py web_port:=5055 rosbridge_port:=9090 start_rosbridge:=true
```

Open:
```text
http://127.0.0.1:5055/rviz3d?frame=tank_map&cloud=detected&rays=1&vectors=1
```

Cloud modes: `off`, `detected`, `all`, `terrain`, `final`, `ground`, `nonground`.

## Scenario2 saved terrain web view

RViz의 `tank_scenario2_map_view.launch.py`와 같은 저장 파일을 web에서 표시한다.

```bash
ros2 launch rviz_web rviz_web_scenario2_map_view.launch.py web_port:=5055 rosbridge_port:=9090 start_rosbridge:=true
```

Open:

```text
http://127.0.0.1:5055/rviz3d?frame=tank_map&cloud=off&rays=0&vectors=0
```

It subscribes to:

- `/tank/rviz/recon_map_markers`
- `/tank/terrain/final_elevation_markers`
- `/tank/terrain/final_wireframe_markers`
- `/tank/terrain/final_ground_points`
- `/tank/terrain/final_non_ground_points`

The final terrain surface is rendered from `visualization_msgs/Marker.TRIANGLE_LIST` with per-vertex colors.


Fix v2: rviz_web_scenario2_map_view.launch.py starts rviz_web_server via `ros2 run rviz_web rviz_web_server` to avoid FileNotFoundError when ExecuteProcess cannot resolve console scripts from PATH.
