FROM pytorch/pytorch:2.9.1-cuda12.6-cudnn9-runtime

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY tracker.py .
COPY models/best.pt /app/models/

RUN mkdir -p uploads outputs

EXPOSE 8000

CMD ["python", "tracker.py"]