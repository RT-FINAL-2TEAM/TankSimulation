#!/usr/bin/env python3
"""독립 실행형 YOLO 디버그 서버용 호환 래퍼.

실제 운영 통합은 이제 ros_bridge /detect 내부에 있다.
독립 실행형 YOLO 테스트는 다음으로 실행한다:
    ros2 run vision yolo_debug_server
또는:
    python3 src/vision/run_yolo_server.py
"""

from vision.yolo_debug_server import main

if __name__ == "__main__":
    main()
