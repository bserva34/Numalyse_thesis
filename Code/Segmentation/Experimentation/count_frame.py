import cv2    

cap = cv2.VideoCapture("test_dataset_V3C1.mp4")
frames = []


while True:
    ret, frame = cap.read()
    if not ret:
        break
    frames.append(frame)

print(len(frames))