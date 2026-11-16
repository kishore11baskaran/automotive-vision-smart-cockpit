
import cv2
from ultralytics import YOLO

# Load YOLO model once
model = YOLO("yolov8n.pt")

# COCO class IDs
TARGET_CLASSES = [0, 1, 2, 3, 5, 7]

VEHICLE_CLASSES = [
    "bicycle",
    "car",
    "motorcycle",
    "bus",
    "truck"
]

# Store unique tracking IDs
tracked_vehicle_ids = set()
tracked_pedestrian_ids = set()


def reset_tracking():
    """Reset IDs when starting a new video."""
    tracked_vehicle_ids.clear()
    tracked_pedestrian_ids.clear()


def detect_objects(frame):
    """
    Detect and track vehicles and pedestrians
    using YOLOv8 and ByteTrack.
    """

    results = model.track(
        frame,
        persist=True,
        tracker="bytetrack.yaml",
        classes=TARGET_CLASSES,
        conf=0.45,
        iou=0.5,
        imgsz=640,
        verbose=False
    )

    annotated_frame = frame.copy()

    vehicle_count = 0
    pedestrian_count = 0

    for result in results:

        for box in result.boxes:

            class_id = int(box.cls[0])
            class_name = model.names[class_id]
            confidence = float(box.conf[0])

            # Get tracking ID
            if box.id is not None:
                track_id = int(box.id[0])
            else:
                track_id = None

            # Bounding box coordinates
            x1, y1, x2, y2 = map(
                int, box.xyxy[0].tolist()
            )

            # Classify and count objects
            if class_name == "person":

                pedestrian_count += 1
                color = (255, 100, 0)

                if track_id is not None:
                    tracked_pedestrian_ids.add(track_id)

            elif class_name in VEHICLE_CLASSES:

                vehicle_count += 1
                color = (0, 200, 0)

                if track_id is not None:
                    tracked_vehicle_ids.add(track_id)

            else:
                continue

            # Draw bounding box
            cv2.rectangle(
                annotated_frame,
                (x1, y1),
                (x2, y2),
                color,
                2
            )

            # Display class, ID and confidence
            if track_id is not None:
                label = (
                    f"ID:{track_id} "
                    f"{class_name} {confidence:.2f}"
                )
            else:
                label = f"{class_name} {confidence:.2f}"

            font = cv2.FONT_HERSHEY_SIMPLEX

            cv2.putText(
                annotated_frame,
                label,
                (x1, max(y1 - 8, 20)),
                font,
                0.5,
                color,
                2,
                cv2.LINE_AA
            )

    # Display current-frame counts
    cv2.putText(
        annotated_frame,
        f"Vehicles: {vehicle_count}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    cv2.putText(
        annotated_frame,
        f"Pedestrians: {pedestrian_count}",
        (20, 75),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 100, 0),
        2
    )

    # Display unique tracked IDs
    cv2.putText(
        annotated_frame,
        f"Unique Vehicles: {len(tracked_vehicle_ids)}",
        (20, 110),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 255, 0),
        2
    )

    cv2.putText(
        annotated_frame,
        f"Unique Pedestrians: {len(tracked_pedestrian_ids)}",
        (20, 140),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 100, 0),
        2
    )

    return (
        annotated_frame,
        vehicle_count,
        pedestrian_count
    )
