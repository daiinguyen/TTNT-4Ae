import cv2, os, torch, numpy as np
from PIL import Image
from torchvision import transforms
from facenet_pytorch import MTCNN, InceptionResnetV1, fixed_image_standardization

# ───────── CONFIG ─────────
IMG_PATH   = './data/input_images/dai.jpg'  # ← ảnh cần nhận diện
DATA_PATH  = './data'                       # chứa faceslist.pth + usernames.npy
THRESH_EUC = 3                              # ngưỡng “unknown” (càng nhỏ càng gắt)
FRAME_SIZE = (640, 480)                     # size gốc của webcam, để crop margin
# ──────────────────────────

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Device:', device)

# Model & detector
embedder = InceptionResnetV1(pretrained='casia-webface').eval().to(device)
mtcnn    = MTCNN(keep_all=True, device=device)

# Load embeddings + tên
embeddings = torch.load(os.path.join(DATA_PATH, 'faceslist.pth'), map_location=device)
names      = np.load(os.path.join(DATA_PATH, 'usernames.npy'))

# Preprocess sang tensor 160×160
to_tensor = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.ToTensor(),
    fixed_image_standardization
])

def extract_face(img, box, margin=20):
    """Crop + resize khuôn mặt từ box (thêm chút margin)."""
    x1,y1,x2,y2 = map(int, box)
    h,w = y2-y1, x2-x1
    margin_x, margin_y = int(margin*w/160), int(margin*h/160)
    x1 = max(x1 - margin_x//2, 0);  y1 = max(y1 - margin_y//2, 0)
    x2 = min(x2 + margin_x//2, FRAME_SIZE[0]);  y2 = min(y2 + margin_y//2, FRAME_SIZE[1])
    face = cv2.resize(img[y1:y2, x1:x2], (160, 160))
    return Image.fromarray(face)

def recognize(face_img):
    """Trả về (label, distance) cho 1 face đã crop."""
    with torch.no_grad():
        emb = embedder(to_tensor(face_img).unsqueeze(0).to(device))   # [1,512]
    diff = emb.unsqueeze(-1) - embeddings.T.unsqueeze(0)              # [1,512,n]
    dist = torch.sum(diff**2, dim=1)                                  # [1,n]
    min_dist, idx = torch.min(dist, dim=1)
    if min_dist.item() > THRESH_EUC:
        return 'Unknown', min_dist.item()
    return names[idx],  min_dist.item()

# ───────── MAIN ─────────
img = cv2.imread(IMG_PATH)
if img is None:
    raise FileNotFoundError(f"Không thấy ảnh: {IMG_PATH}")

img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
boxes, _ = mtcnn.detect(img_rgb)

if boxes is not None:
    for box in boxes:
        face = extract_face(img, box)
        label, dist = recognize(face)

        x1,y1,x2,y2 = map(int, box)
        color = (0,255,0) if label != 'Unknown' else (0,0,255)
        cv2.rectangle(img, (x1,y1), (x2,y2), color, 2)
        cv2.putText(img, str(label), (x1, y1-10),
                    cv2.FONT_HERSHEY_DUPLEX, 1, color, 2)
else:
    print("Không phát hiện khuôn mặt nào 👀")

cv2.imshow("Face Recognition – Image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
