import cv2
import numpy as np
import time
import os
import tkinter as tk
from PIL import Image, ImageTk

# -------------------- FOLDERS --------------------
os.makedirs("recordings", exist_ok=True)

# -------------------- LOAD YOLO --------------------
net = cv2.dnn.readNet("yolo/yolov3.weights", "yolo/yolov3.cfg")

with open("yolo/coco.names") as f:
    classes = f.read().strip().split("\n")

# Classroom object filter
classroom_objects = {
    "person","chair","bench","desk","table","laptop","book",
    "cell phone","backpack","bottle","clock","tvmonitor","keyboard","mouse",
    "fan", "tubelight"
}

# Optional: map COCO classes to fan/tubelight for quick demo
fake_map = {
    "clock": "tubelight",
    "tvmonitor": "fan"
}

layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# -------------------- CAMERA --------------------
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

fourcc = cv2.VideoWriter_fourcc(*'XVID')
video_name = f"recordings/detection_{int(time.time())}.avi"
out = cv2.VideoWriter(video_name, fourcc, 20.0, (640, 480))

running = False
prev_time = 0

# -------------------- FUNCTIONS --------------------
def start_detection():
    global running
    if not running:
        running = True
        video_loop()

def stop_detection():
    global running
    running = False

def video_loop():
    global running, prev_time

    if not running:
        return

    ret, frame = cap.read()
    if not ret:
        return

    height, width, _ = frame.shape

    # YOLO preprocessing
    blob = cv2.dnn.blobFromImage(frame, 1/255, (416,416), (0,0,0), True, crop=False)
    net.setInput(blob)
    outputs = net.forward(output_layers)

    boxes, confidences, class_ids = [], [], []
    object_count = {}

    # MULTI OBJECT DETECTION
    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > 0.5:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    total_objects = 0

    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = classes[class_ids[i]]

            # Apply fake mapping for quick demo
            if label in fake_map:
                label = fake_map[label]

            confidence = confidences[i]

            # Only classroom objects
            if label not in classroom_objects:
                continue

            total_objects += 1
            object_count[label] = object_count.get(label, 0) + 1

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f"{label} {confidence:.2f}",
                        (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # FPS calculation
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time + 0.0001)
    prev_time = curr_time

    cv2.putText(frame, f"FPS: {int(fps)}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    cv2.putText(frame, f"Objects: {total_objects}", (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    out.write(frame)

    # Object Counter Panel
    text = "  ".join([f"{k}: {v}" for k, v in object_count.items()])
    counter_label.config(text=text if text else "No classroom objects detected")

    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)
    imgtk = ImageTk.PhotoImage(image=img)

    video_label.imgtk = imgtk
    video_label.configure(image=imgtk)

    root.after(1, video_loop)

def on_close():
    global running
    running = False
    cap.release()
    out.release()
    root.destroy()

# -------------------- GUI --------------------
root = tk.Tk()
root.title("AI Smart Classroom Detection System")
root.geometry("1100x800")
root.configure(bg="#0d1117")

title = tk.Label(root, text="AI Smart Classroom Detection System",
                 font=("Segoe UI", 26, "bold"),
                 bg="#0d1117", fg="#00ff99")
title.pack(pady=15)

video_label = tk.Label(root, bg="#0d1117", bd=3, relief="ridge")
video_label.pack(pady=10)

counter_label = tk.Label(root, text="No classroom objects detected",
                         font=("Segoe UI", 14),
                         bg="#0d1117", fg="#58a6ff")
counter_label.pack(pady=10)

btn_frame = tk.Frame(root, bg="#0d1117")
btn_frame.pack(pady=20)

start_btn = tk.Button(btn_frame, text="▶ Start Detection",
                      font=("Segoe UI", 15, "bold"),
                      bg="#00ff99", fg="black",
                      width=16, height=2,
                      command=start_detection)
start_btn.grid(row=0, column=0, padx=15)

stop_btn = tk.Button(btn_frame, text="⏹ Stop",
                     font=("Segoe UI", 15, "bold"),
                     bg="#ff4444", fg="white",
                     width=16, height=2,
                     command=stop_detection)
stop_btn.grid(row=0, column=1, padx=15)

status = tk.Label(root, text="Smart Classroom Mode • Auto Recording • Multi Object Detection • FPS Monitor",
                  font=("Segoe UI", 12),
                  bg="#0d1117", fg="gray")
status.pack()

root.protocol("WM_DELETE_WINDOW", on_close)
root.mainloop()
