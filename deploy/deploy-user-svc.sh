#!/bin/bash
WORK_DIR="/opt/microservice-project"
TAR_FILE="${WORK_DIR}/user-extra-svc.tar"
MANIFEST_DIR="${WORK_DIR}/microservices-demo/kubernetes-manifests"

# 切换到 Minikube 内置Docker（离线必需）
eval $(minikube docker-env)

# 导入离线镜像
if [ ! -f ${TAR_FILE} ];then
    echo "❌ 未找到镜像包 user-extra-svc.tar"
    exit 1
fi
docker load -i ${TAR_FILE}
echo "✅ 镜像导入成功"

# 部署K8s资源
kubectl apply -f ${MANIFEST_DIR}/user-extra-svc.yaml
echo "✅ K8s 资源部署完成"

# 查看运行状态
echo -e "\n==== Pod 状态 ===="
kubectl get pods | grep user-extra-svc

echo -e "\n==== Service 状态 ===="
kubectl get svc | grep user-extra-svc

# 容器内接口测试
echo -e "\n==== 接口测试 ===="
echo "1. 健康接口："
kubectl exec deploy/user-extra-svc -- curl -s http://127.0.0.1:8080

echo -e "\n2. 查询用户(10086)："
kubectl exec deploy/user-extra-svc -- curl -s http://127.0.0.1:8080/api/user/info/10086

echo -e "\n3. 扣除积分(100)："
kubectl exec deploy/user-extra-svc -- curl -s http://127.0.0.1:8080/api/user/credit/deduct/10086/100
