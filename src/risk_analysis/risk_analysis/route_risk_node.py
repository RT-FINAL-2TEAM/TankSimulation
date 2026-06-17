import json
from pathlib import Path

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

from .llm_reporter import LLMReporter
from .route_risk_io import load_json, save_json


class RouteRiskNode(Node):
    def __init__(self):
        super().__init__("route_risk_node")

        self.declare_parameter("input_path", "recon_reports/route_comparison.json")
        self.declare_parameter("output_path", "recon_reports/route_risk_result.json")
        self.declare_parameter("ollama_url", "http://localhost:11434/api/generate")
        self.declare_parameter("model_name", "qwen3:0.6b")
        self.declare_parameter("mode", "file_once")

        self.input_path = Path(
            self.get_parameter("input_path").get_parameter_value().string_value
        )
        self.output_path = Path(
            self.get_parameter("output_path").get_parameter_value().string_value
        )
        self.mode = self.get_parameter("mode").get_parameter_value().string_value

        ollama_url = self.get_parameter("ollama_url").get_parameter_value().string_value
        model_name = self.get_parameter("model_name").get_parameter_value().string_value

        self.llm_reporter = LLMReporter(
            ollama_url=ollama_url,
            model_name=model_name,
        )

        self.result_pub = self.create_publisher(
            String,
            "/tank/risk/route_report",
            10,
        )

        self.subscription = self.create_subscription(
            String,
            "/tank/risk/route_comparison",
            self.on_route_comparison,
            10,
        )

        self.get_logger().info("route_risk_node started")
        self.get_logger().info(f"mode={self.mode}")
        self.get_logger().info(f"input_path={self.input_path}")
        self.get_logger().info(f"output_path={self.output_path}")

        if self.mode == "file_once":
            self.run_from_file_once()

    def run_from_file_once(self):
        try:
            comparison_data = load_json(self.input_path)
            result = self.llm_reporter.generate_route_decision(comparison_data)

            result["input_file"] = str(self.input_path)
            result["output_file"] = str(self.output_path)

            saved_path = save_json(self.output_path, result)
            result["saved_path"] = saved_path

            self.publish_result(result)

            self.get_logger().info(f"risk result saved: {saved_path}")

        except Exception as e:
            self.get_logger().error(f"file_once analysis failed: {e}")

    def on_route_comparison(self, msg: String):
        try:
            comparison_data = json.loads(msg.data)
            result = self.llm_reporter.generate_route_decision(comparison_data)

            result["source"] = "topic"
            result["output_file"] = str(self.output_path)

            saved_path = save_json(self.output_path, result)
            result["saved_path"] = saved_path

            self.publish_result(result)

            self.get_logger().info("risk result generated from topic")

        except Exception as e:
            self.get_logger().error(f"topic analysis failed: {e}")

    def publish_result(self, result: dict):
        msg = String()
        msg.data = json.dumps(result, ensure_ascii=False)
        self.result_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)

    node = RouteRiskNode()

    try:
        if node.mode == "file_once":
            rclpy.spin_once(node, timeout_sec=0.5)
        else:
            rclpy.spin(node)

    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()