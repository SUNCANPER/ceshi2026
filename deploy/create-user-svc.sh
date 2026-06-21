#!/bin/bash
# 基础路径
WORK_DIR="/opt/microservice-project"
SVC_NAME="user-extra-svc"
SVC_DIR="${WORK_DIR}/${SVC_NAME}"
MANIFEST_DIR="${WORK_DIR}/microservices-demo/kubernetes-manifests"

# 创建服务目录
mkdir -p ${SVC_DIR}
cd ${SVC_DIR}

# 业务代码 app.py
cat > app.py << 'APP_EOF'
from flask import Flask
import os

app = Flask(__name__)
SERVICE_NAME = os.getenv("SERVICE_NAME", "user-extra-svc")

# 模拟用户数据
user_data = {
    "10086": {"level": "VIP", "credit": 888, "register_days": 365},
    "10001": {"level": "普通会员", "credit": 200, "register_days": 90},
    "10002": {"level": "VIP", "credit": 1200, "register_days": 500}
}

# 健康检查接口
@app.route("/health")
def health():
    return {"status": "ok"}

# 用户信息查询接口
@app.route("/api/user/info/<uid>")
def user_info(uid):
    if uid not in user_data:
        return {"code": 404, "msg": "用户不存在"}, 404
    data = user_data[uid]
    return {
        "service": SERVICE_NAME,
        "user_id": uid,
        "level": data["level"],
        "credit": data["credit"],
        "register_days": data["register_days"]
    }

# 积分扣除接口
@app.route("/api/user/credit/deduct/<uid>/<num>")
def deduct_credit(uid, num):
    if uid not in user_data:
        return {"code": 404, "msg": "用户不存在"}, 400
    try:
        deduct_num = int(num)
    except:
        return {"code": 400, "msg": "扣除积分必须为数字"}, 400
    current_credit = user_data[uid]["credit"]
    if deduct_num > current_credit:
        return {"code": 400, "msg": "积分不足，无法扣除"}, 400
    user_data[uid]["credit"] -= deduct_num
    return {
        "service": SERVICE_NAME,
        "user_id": uid,
        "deduct_num": deduct_num,
        "remaining_credit": user_data[uid]["credit"],
        "status": "扣除成功"
    }

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
APP_EOF

# Dockerfile
cat > Dockerfile << 'DOCKER_EOF'
FROM python:3.9-slim
WORKDIR /app
COPY . .
RUN pip install flask -i https://pypi.tuna.tsinghua.edu.cn/simple
ENV OTEL_EXPORTER_JAEGER_AGENT_HOST=jaeger-all-in-one.observability.svc.cluster.local
ENV OTEL_EXPORTER_JAEGER_AGENT_PORT=6831
EXPOSE 8080
CMD ["python", "app.py"]
DOCKER_EOF

# K8s 部署清单（默认ClusterIP，后续改NodePort）
cat > ${MANIFEST_DIR}/user-extra-svc.yaml << 'YAML_EOF'
apiVersion: apps/v1
kind: Deployment
metadata:
  name: user-extra-svc
  labels:
    app: user-extra-svc
spec:
  replicas: 1
  selector:
    matchLabels:
      app: user-extra-svc
  template:
    metadata:
      labels:
        app: user-extra-svc
    spec:
      containers:
      - name: user-extra
        image: user-extra-svc:v1
        imagePullPolicy: Never
        ports:
        - containerPort: 8080
        readinessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 5
          periodSeconds: 10
---
apiVersion: v1
kind: Service
metadata:
  name: user-extra-svc
spec:
  selector:
    app: user-extra-svc
  ports:
  - port: 8080
    targetPort: 8080
YAML_EOF

echo "====================================="
echo "✅ 代码、Dockerfile、K8s配置生成完成"
echo "====================================="
