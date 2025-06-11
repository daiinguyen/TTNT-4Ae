## **Quick Start**
This is a FaceNet and MTCNN inference from [this Repository](https://github.com/timesler/facenet-pytorch).
1. Install:  
    Clone source code và cài thư viện.  
    
    ```bash
    # Clone Repo:
<<<<<<< HEAD
    git clone https://github.com/daiinguyen/TTNT-4Ae
=======
    git clone https://github.com/daiinguyen/TTNT-4Ae.git
>>>>>>> c60d988131312f25bd8b52254d5f96e86c3e83b7
    
    # Install with Pip
    pip install -r requirements.txt

    ```
<<<<<<< HEAD
1. Detection & Capturing:
    Chạy file face_cap.py để chụp 1 ảnh test  
    Chạy file face_capture.py để chụp ảnh lấy dữ liệu training  
  

=======
1. Detection & Capturing:  
    Chạy file face_cap.py để chụp 1 ảnh test  
    Chạy file face_capture.py để chụp ảnh lấy dữ liệu training  
    
>>>>>>> c60d988131312f25bd8b52254d5f96e86c3e83b7
    ```bash
    # Face Detection:
    python face_detect.py
    
    # Face Capturing (Remember to input your name FIRST in console):
    python face_capture.py

    ```
<<<<<<< HEAD
1. Create FaceList and Recognition:  
=======
1. Create FaceList and Recognition:
>>>>>>> c60d988131312f25bd8b52254d5f96e86c3e83b7
    Chạy file update_faces.py để update dữ liệu  
    Chạy file face_recognition.py để chạy chương trình nhận diện realtime trên webcam  
    Chạy file recog_image.py để nhận nhiện trên ảnh (Lưu ý đổi đường dẫn đến ảnh muốn test)  
    ```bash
    # Update FaceList:
    python update_faces.py
    
    # Face Recognition:
    python recog_image.py
    python face_recognition.py

    ```
