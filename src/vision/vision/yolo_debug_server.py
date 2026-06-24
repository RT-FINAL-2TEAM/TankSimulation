# -*- coding: utf-8 -*-
"""Tank Challenge YOLO 디텍터용 독립 실행 디버그 서버.

ros_bridge와 무관하게 모델만 단독으로 테스트할 때만 쓴다:
    ros2 run vision yolo_debug_server

port 5000의 ros_bridge와 충돌하지 않도록 의도적으로 기본 포트를 5055로 둔다.
"""

import os
from flask import Flask, jsonify, request

from .yolo_detector import get_detector

app = Flask(__name__)


@app.route('/detect', methods=['POST'])
def detect():
    image = request.files.get('image')
    if image is None:
        return jsonify({"error": "No image received"}), 400
    detections = get_detector().detect_bytes(image.read())
    return jsonify(detections)


@app.route('/debug/yolo', methods=['GET'])
def debug_yolo():
    return jsonify(get_detector().debug_state())


def main() -> None:
    host = os.getenv('YOLO_DEBUG_HOST', '0.0.0.0')
    port = int(os.getenv('YOLO_DEBUG_PORT', '5055'))
    print(f"Tank YOLO debug server: http://{host}:{port}")
    app.run(host=host, port=port, threaded=True)


if __name__ == '__main__':
    main()
