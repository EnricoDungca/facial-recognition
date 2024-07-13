from ultralytics import YOLO
import cv2
import os


model = YOLO(r"Z:\Face_Recognition\Another try\model\yolov8n.pt")  # load model

video_path = r"Z:\Face_Recognition\cctvVideo\2024-07-05 11-07-51.mp4"
cap = cv2.VideoCapture(video_path)

ret = True
frame_count = 0  # Initialize a frame counter
save_dir = r"Z:\School\Workspace_Py\activityDetection\FrameByFrame"  # Directory to save the crops
os.makedirs(save_dir, exist_ok=True)  # Create the directory if it doesn't exist

while ret:
    ret, frame = cap.read()
    frame_count += 1  # Increment the frame counter

    # Check if the frame was successfully read
    if ret:
        
        # Perform object detection
        results = model.track(frame, persist=True)

        # Check if results is not None before accessing results[0].boxes
        if results is not None and results[0].boxes is not None:
            # Iterate over the detected objects 
            for i, det in enumerate(results[0].boxes.xyxy):
                # Extract bounding box coordinates
                x1, y1, x2, y2 = map(int, det[:4])
                crop = frame[y1:y2, x1:x2]  # Crop the detected object
                cv2.imwrite(os.path.join(save_dir, f"frame_{frame_count}_{i}.jpg"), crop)
                

                

        frame_ = results[0].plot() if results is not None else None

        cv2.imshow("frame", frame_)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()