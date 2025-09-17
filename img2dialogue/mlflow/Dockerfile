FROM python:3.11-alpine
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install -r requirements.txt
EXPOSE 5000
WORKDIR /mlflow
CMD ["mlflow", "server", "--host", "0.0.0.0", "--port", "5000"]