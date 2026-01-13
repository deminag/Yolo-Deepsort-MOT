from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
import uvicorn
import os
import uuid
import cv2
from ultralytics import YOLO
import numpy as np

app = FastAPI()

# Загружаем модель
model = YOLO(r"C:\Users\bunny\Desktop\SORT-DeepSORT-Tracker-main\runs\detect\100\best.pt")

@app.post("/track")
async def track_video(
    file: UploadFile = File(...),
    conf: float = 0.25
):
    
    # Сохраняем видео
    video_id = str(uuid.uuid4())[:8]
    input_path = f"uploads/{video_id}.mp4"
    output_path = f"outputs/tracked_{video_id}.mp4"
    
    # Сохраняем загруженный файл
    with open(input_path, "wb") as f:
        content = await file.read()
        f.write(content)
    
    print(f"Обрабатываю видео: {input_path}")
    
    # Открываем видео
    cap = cv2.VideoCapture(input_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Создаем выходное видео
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Детекция и трекинг
        results = model.track(frame, conf=conf, persist=True, verbose=False)
        
        # Рисуем результаты
        if results[0].boxes is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            if results[0].boxes.id is not None:
                track_ids = results[0].boxes.id.cpu().numpy().astype(int)
            
            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = map(int, box)
                
                color = (0, 255, 0)
                
                if results[0].boxes.id is not None and i < len(track_ids):
                    track_id = track_ids[i]
                    # Генерируем цвет и преобразуем в int
                    r = int((track_id * 50) % 255)
                    g = int((track_id * 100) % 255)
                    b = int((track_id * 150) % 255)
                    # OpenCV использует BGR, поэтому порядок: (синий, зеленый, красный)
                    color = (b, g, r)
                
                # Рисуем бокс
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                
                # Подпись
                if results[0].boxes.id is not None and i < len(track_ids):
                    track_id = track_ids[i]
                    label = f"ID: {track_id}"
                    cv2.putText(frame, label, (x1, y1-10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Сохраняем кадр
        out.write(frame)
        frame_count += 1
        
        # Показываем прогресс каждые 10 кадров
        if frame_count % 100 == 0:
            print(f"Обработано кадров: {frame_count}")
    
    # Закрываем видео
    cap.release()
    out.release()
    
    print(f"Обработано {frame_count} кадров")
    
    # Возвращаем результат
    return FileResponse(
        path=output_path,
        filename=f"tracked_{file.filename}",
        media_type="video/mp4"
    )

@app.get("/")
def home():
    return {"message": "YOLO Video Tracker", "endpoint": "POST /track"}

if __name__ == "__main__":
    os.makedirs("uploads", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)
    uvicorn.run(app, host="0.0.0.0", port=8000)