import cv2
import os

SAVE_PATH = './data/input_images'


# Nhập tên người dùng
name = input("Nhập tên của bạn (không dấu, không khoảng trắng): ").strip()
if name == '':
    print("Tên không hợp lệ. Dừng chương trình.")
    exit()

IMG_NAME = f"{name}.jpg"
IMG_FULL_PATH = os.path.join(SAVE_PATH, IMG_NAME)

# Mở webcam
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print("\n>>> Nhấn 'SPACE' để chụp và lưu ảnh.")
print(">>> Nhấn 'ESC' để thoát mà không lưu.\n")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Không lấy được frame từ webcam.")
        break

    # Hiển thị frame
    cv2.imshow("Chup anh", frame)
    key = cv2.waitKey(1)

    if key == 27:  # ESC
        print("Thoát không chụp.")
        break
    elif key == 32:  # SPACE
        cv2.imwrite(IMG_FULL_PATH, frame)
        print(f"✅ Đã lưu ảnh: {IMG_FULL_PATH}")
        break

cap.release()
cv2.destroyAllWindows()
