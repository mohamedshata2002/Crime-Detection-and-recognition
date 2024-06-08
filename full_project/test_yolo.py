import cv2
from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO("E:\\workstation\\project\\project\\yolo\\best.pt")

# Open the video file
video_path = "E:\\workstation\\project\\project\\fire.mp4"
cap = cv2.VideoCapture(video_path)
names = model.names
# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        # Run YOLOv8 tracking on the frame, persisting tracks between frames
        results = model.track(frame, persist=True)

        # Visualize the results on the frame
        annotated_frame = results[0].plot()

        # Display the annotated frame
        cv2.imshow("YOLOv8 Tracking", annotated_frame)
        for m in results[0].boxes:
            cls = m.cls
            conf =m.conf.numpy()[0]
            if conf >= 0.68:
                 print(names[int(cls)],conf)
        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()
