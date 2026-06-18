# -*- coding: utf-8 -*-
"""Standalone debug server for the Tank Challenge YOLO detector.

Use this only for testing the model independent of ros_bridge:
    ros2 run vision yolo_debug_server

It intentionally defaults to port 5055 so it does not conflict with ros_bridge on port 5000.
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
