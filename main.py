# /home/shubhamos/.pyenv/versions/emotion-env/bin/python main.py

import cv2
import numpy as np
import threading
import time
import os
from fer import FER
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

app = FastAPI()

# Global state
class GlobalState:
    def __init__(self):
        self.frame = None
        self.detections = []
        self.ai_active = True
        self.lock = threading.Lock()
        self.running = True

state = GlobalState()

# 1. Background Camera Thread (Fast & Smooth)
def camera_worker():
    print("[Camera] Opening webcam...")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[Camera] ERROR: Could not open camera.")
        return

    while state.running:
        ret, frame = cap.read()
        if not ret:
            cap.release()
            time.sleep(1)
            cap = cv2.VideoCapture(0)
            continue
        with state.lock:
            state.frame = frame.copy()
        time.sleep(0.01)

# 2. Background AI Thread (Multi-User Robust)
def ai_worker():
    print("[AI] Initializing Multi-User FER Engine...")
    try:
        # Use MTCNN=True for high accuracy if possible, but it might be slow.
        # Let's stick with False for broad compatibility but improve detection params.
        detector = FER(mtcnn=False) 
        print("[AI] Engine Ready.")
    except Exception as e:
        print(f"[AI] CRITICAL: {e}")
        return

    while state.running:
        if not state.ai_active:
            with state.lock:
                state.detections = []
            time.sleep(0.5)
            continue

        local_frame = None
        with state.lock:
            if state.frame is not None:
                local_frame = state.frame.copy()
        
        if local_frame is not None:
            try:
                # Detect emotions for ALL faces
                results = detector.detect_emotions(local_frame)
                
                new_detections = []
                for face in results:
                    box = face["box"]
                    # Filter out 'neutral' and 'fear' and convert to standard floats
                    raw_emotions = face["emotions"]
                    emotions_dict = {k: float(v) for k, v in raw_emotions.items() if k not in ['neutral', 'fear']}
                    
                    # Compute dominant from remaining
                    if emotions_dict:
                        dominant = max(emotions_dict, key=emotions_dict.get)
                    else:
                        dominant = "unknown"
                    
                    new_detections.append({
                        "box": [int(box[0]), int(box[1]), int(box[2]), int(box[3])],
                        "emotions": emotions_dict,
                        "dominant": dominant
                    })

                with state.lock:
                    state.detections = new_detections
            except Exception as e:
                print(f"[AI] Error: {e}")

        # Decoupled processing speed
        time.sleep(0.1)

threading.Thread(target=camera_worker, daemon=True).start()
threading.Thread(target=ai_worker, daemon=True).start()

@app.get('/video_feed')
async def video_feed():
    def frame_generator():
        while state.running:
            local_frame = None
            local_detections = []
            with state.lock:
                if state.frame is not None:
                    local_frame = state.frame.copy()
                    local_detections = list(state.detections)
            
            if local_frame is not None:
                for det in local_detections:
                    bx = det["box"]
                    # Neon branding colors
                    color = (56, 189, 248) # Default Blue
                    dom = det["dominant"]
                    if dom == 'happy': color = (34, 197, 94)
                    elif dom in ['angry', 'disgust']: color = (239, 68, 68)
                    
                    cv2.rectangle(local_frame, (bx[0], bx[1]), (bx[0]+bx[2], bx[1]+bx[3]), color, 2)
                    label = f"{dom.upper()} {int(det['emotions'][dom]*100)}%"
                    cv2.putText(local_frame, label, (bx[0], bx[1]-10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

                _, buffer = cv2.imencode('.jpg', local_frame)
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            time.sleep(0.04)
    return StreamingResponse(frame_generator(), media_type='multipart/x-mixed-replace; boundary=frame')

@app.get('/emotions')
async def get_emotions():
    with state.lock:
        return {"active": state.ai_active, "detections": state.detections}

@app.post('/toggle_ai')
async def toggle_ai():
    with state.lock:
        state.ai_active = not state.ai_active
        return {"status": "ok", "active": state.ai_active}

# App Setup
os.makedirs("templates", exist_ok=True)
os.makedirs("static", exist_ok=True)
if os.path.exists("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, access_log=False)
