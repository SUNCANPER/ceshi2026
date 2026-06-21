FROM python:3.9-slim
WORKDIR /app
COPY . .
RUN pip install flask -i https://pypi.tuna.tsinghua.edu.cn/simple
ENV OTEL_EXPORTER_JAEGER_AGENT_HOST=jaeger-all-in-one.observability.svc.cluster.local
ENV OTEL_EXPORTER_JAEGER_AGENT_PORT=6831
EXPOSE 8080
CMD ["python", "app.py"]
