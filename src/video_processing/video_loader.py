
import cv2
import time

from src.object_detection.detector import detect_objects

# Video path
video_path = "data/sample_videos/road_traffic.mp4"

# Open video
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

print("Video opened successfully!")

# FPS calculation variables
previous_time = time.time()
fps = 0

while True:

    ret, frame = cap.read()

    if not ret:
        break

    # Detect vehicles and pedestrians
    detected_frame, vehicles, pedestrians = detect_objects(frame)

    # Calculate FPS
    current_time = time.time()
    elapsed_time = current_time - previous_time

    if elapsed_time > 0:
        fps = 1 / elapsed_time

    previous_time = current_time

    # Get frame dimensions
    height, width = detected_frame.shape[:2]

    # Information panel
    panel_width = 300
    panel_height = 125

    cv2.rectangle(
        detected_frame,
        (10, 10),
        (panel_width, panel_height),
        (25, 25, 25),
        -1
    )

    # Application title
    cv2.putText(
        detected_frame,
        "AUTOMOTIVE SMART VISION",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1
    )

    # Vehicle count
    cv2.putText(
        detected_frame,
        f"Vehicles: {vehicles}",
        (20, 65),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 255, 0),
        2
    )

    # Pedestrian count
    cv2.putText(
        detected_frame,
        f"Pedestrians: {pedestrians}",
        (20, 92),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 150, 0),
        2
    )

    # FPS counter
    cv2.putText(
        detected_frame,
        f"FPS: {fps:.1f}",
        (20, 115),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 255, 255),
        1
    )

    # Display video
    cv2.imshow(
        "Automotive Smart Vision",
        detected_frame
    )

    # Press Q to exit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()

print("Video processing stopped.")
