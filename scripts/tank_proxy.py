from flask import Flask, request, Response
import requests
import os
from pathlib import Path

app = Flask(__name__)

def load_env_file():
    """
    현재 proxy_server.py와 같은 폴더의 .env 파일을 읽어서 os.environ에 반영한다.
    이미 등록된 환경변수는 덮어쓰지 않는다.
    """
    env_path = Path(__file__).resolve().parent / ".env"

    if not env_path.exists():
        print(f"[ENV] .env file not found: {env_path}")
        return

    with env_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if not line or line.startswith("#"):
                continue

            if "=" not in line:
                continue

            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")

            os.environ.setdefault(key, value)

    print(f"[ENV] loaded: {env_path}")


load_env_file()

UBUNTU_SERVER = os.environ.get("UBUNTU_SERVER", "http://127.0.0.1:5005")
PROXY_HOST = os.environ.get("PROXY_HOST", "127.0.0.1")
PROXY_PORT = int(os.environ.get("PROXY_PORT", "5005"))

# UBUNTU_SERVER = "http://172.27.119.129:5005"  # Ubuntu PC IP로 수정

@app.route("/", defaults={"path": ""}, methods=["GET", "POST", "PUT", "DELETE"])
@app.route("/<path:path>", methods=["GET", "POST", "PUT", "DELETE"])
def proxy(path):
    target_url = f"{UBUNTU_SERVER}/{path}"

    try:
        if request.method == "GET":
            resp = requests.get(
                target_url,
                params=request.args,
                timeout=10
            )

        elif request.method == "POST":
            files = {}
            for key, file in request.files.items():
                files[key] = (
                    file.filename,
                    file.stream,
                    file.content_type
                )

            data = request.form.to_dict()
            json_data = request.get_json(silent=True)

            if files:
                resp = requests.post(
                    target_url,
                    files=files,
                    data=data,
                    timeout=30
                )
            elif json_data is not None:
                resp = requests.post(
                    target_url,
                    json=json_data,
                    timeout=10
                )
            else:
                resp = requests.post(
                    target_url,
                    data=request.get_data(),
                    headers={
                        "Content-Type": request.headers.get("Content-Type", "")
                    },
                    timeout=10
                )

        else:
            return Response("Unsupported method", status=405)

        return Response(
            resp.content,
            status=resp.status_code,
            content_type=resp.headers.get("Content-Type")
        )

    except Exception as e:
        return Response(f"Proxy error: {e}", status=500)


if __name__ == "__main__":
    print(f"[PROXY] listening on http://{PROXY_HOST}:{PROXY_PORT}")
    print(f"[PROXY] forwarding to {UBUNTU_SERVER}")
    app.run(host=PROXY_HOST, port=PROXY_PORT)
