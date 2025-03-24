import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import threading
import h5py
import json

cap = cv2.VideoCapture(6)

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_draw = mp.solutions.drawing_utils

c_obj = {'Orthogonal': tf.keras.initializers.Orthogonal}

def load_model(f):
    with h5py.File(f, 'r') as h:
        cfg = json.loads(h.attrs['model_config'])
        for l in cfg['config']['layers']:
            l['config'].pop('time_major', None)
        m = tf.keras.models.model_from_json(json.dumps(cfg), custom_objects=c_obj)
        m.set_weights([h['model_weights'][n][()] for n in h['model_weights'].attrs['weight_names']])
    return m

model = load_model("lstm-model.h5")

def get_lm(r):
    return [l for lm in r.pose_landmarks.landmark for l in [lm.x, lm.y, lm.z, lm.visibility]] if r.pose_landmarks else []

def draw(frame, r, lbl):
    mp_draw.draw_landmarks(frame, r.pose_landmarks, mp_pose.POSE_CONNECTIONS)
    x_min, y_min = min([lm.x for lm in r.pose_landmarks.landmark]), min([lm.y for lm in r.pose_landmarks.landmark])
    x_max, y_max = max([lm.x for lm in r.pose_landmarks.landmark]), max([lm.y for lm in r.pose_landmarks.landmark])
    h, w, _ = frame.shape
    cv2.rectangle(frame, (int(x_min * w), int(y_min * h)), (int(x_max * w), int(y_max * h)), (0, 255, 0), 2)
    cv2.putText(frame, f"Status: {lbl}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0) if lbl == "neutral" else (0, 0, 255), 2)
    return frame

def detect(m, lm):
    global lbl
    if len(lm) < 20:
        return
    res = m.predict(np.expand_dims(np.array(lm), 0), verbose=0)[0]
    lbl = "violent" if res[0] > 0.5 else "neutral"

cv2.namedWindow("Pose Tracking", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Pose Tracking", 1200, 1000)

i, warm_up, lbl = 0, 60, "neutral"
lm_list = []

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    r = pose.process(frame_rgb)
    i += 1
    if i > warm_up and r.pose_landmarks:
        lm = get_lm(r)
        lm_list.append(lm)
        if len(lm_list) == 20:
            threading.Thread(target=detect, args=(model, lm_list), daemon=True).start()
            lm_list = []
        frame = draw(frame, r, lbl)
    cv2.imshow("Pose Tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
