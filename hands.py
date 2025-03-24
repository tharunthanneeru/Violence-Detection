import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import threading
import h5py
import json

cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(1, 0.5, 0.5)
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

model = load_model("lstm-hand-gripping.h5")

def get_lm(r):
    return [lm for h in r.multi_hand_landmarks for lm in [l.x for l in h.landmark] + [l.y for l in h.landmark] + [l.z for l in h.landmark]] if r.multi_hand_landmarks else []

def draw(frame, r, lbl):
    for h in r.multi_hand_landmarks:
        mp_draw.draw_landmarks(frame, h, mp_hands.HAND_CONNECTIONS)
        x_min, y_min = min([l.x for l in h.landmark]), min([l.y for l in h.landmark])
        x_max, y_max = max([l.x for l in h.landmark]), max([l.y for l in h.landmark])
        h, w, _ = frame.shape
        cv2.rectangle(frame, (int(x_min * w), int(y_min * h)), (int(x_max * w), int(y_max * h)), (0, 255, 0), 2)
        cv2.putText(frame, f"Status: {lbl}", (int(x_min * w), int(y_max * h) + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
    return frame

def detect(m, lm):
    global lbl
    if len(lm) < 20:
        return
    res = m.predict(np.expand_dims(np.array(lm), 0), verbose=0)[0]
    lbl = "grasped" if any(res[i] > 0.5 for i in [1, 2, 3]) else "not grasped"

cv2.namedWindow("Hand Tracking", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Hand Tracking", 1200, 1000)

i, warm_up, lbl = 0, 60, "not grasped"
lm_list = []

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    r = hands.process(frame_rgb)
    i += 1
    if i > warm_up and r.multi_hand_landmarks:
        lm = get_lm(r)
        lm_list.append(lm)
        if len(lm_list) == 20:
            threading.Thread(target=detect, args=(model, lm_list), daemon=True).start()
            lm_list = []
        frame = draw(frame, r, lbl)
    cv2.imshow("Hand Tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
