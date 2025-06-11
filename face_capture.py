import cv2
from facenet_pytorch import MTCNN
import torch
import os

# ─── 1. Device ─────────────────────────────────────────────────────────
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Running on:', device)

# ─── 2. Paths & user stuff ──────────────────────────────────────────────
IMG_PATH = './data/test_images/'
usr_name = input("Input ur name: ")
USR_PATH = os.path.join(IMG_PATH, usr_name)
os.makedirs(USR_PATH, exist_ok=True)   # ensure folder exists

# ─── 3. Capture params ──────────────────────────────────────────────────
num_to_save = 30      # how many shots you want
skip_every = 2        # save every other frame (∵ leap%2 in your OG code)

# MTCNN just for detection check (no cropping)
mtcnn = MTCNN(keep_all=False, device=device)

# ─── 4. Video loop ──────────────────────────────────────────────────────
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

frame_idx = 0
while cap.isOpened() and num_to_save:
    ret, frame = cap.read()
    if not ret:
        break

    # Face detection → returns None if no face
    boxes, _ = mtcnn.detect(frame)

    # If a face is present AND we're on a “keep” frame, save full frame
    if boxes is not None and len(boxes) and frame_idx % skip_every == 0:
        save_path = os.path.join(USR_PATH, f'0{num_to_save}.jpg')
        cv2.imwrite(save_path, frame)
        print(f'Saved: {save_path}')
        num_to_save -= 1

    frame_idx += 1
    cv2.imshow('Face Capturing', frame)

    # ESC key to bail
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
