
import cv2
from pathlib import Path

from src.object_detection.detector import (
    detect_objects,
    reset_tracking
)


def main():

    # Find project root
    project_root = Path(__file__).resolve().parents[2]

    # Video location
    video_path = (
        project_root
        / "data"
        / "sample_videos"
        / "road_traffic.mp4"
    )

    # Open video
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        print("Error: Could not open video!")
        return

    print("Video opened successfully!")

    # Reset tracking IDs for a new video
    reset_tracking()

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)

    if fps <= 0:
        fps = 30

    delay = max(1, int(1000 / fps))

    # Process video frames
    while True:

        ret, frame = cap.read()

        if not ret:
            print("Video processing completed!")
            break

        # Detect and track objects
        annotated_frame, vehicle_count, pedestrian_count = (
            detect_objects(frame)
        )

        # Display output
        cv2.imshow(
            "Automotive Smart Vision",
            annotated_frame
        )

        # Press Q to exit
        if cv2.waitKey(delay) & 0xFF == ord("q"):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()

    print("Application closed successfully!")


if __name__ == "__main__":
    main()
