import cv2
import mediapipe as mp


class PoseEstimator:
    """
    MediaPipe Pose wrapper (webcam-safe).

    Returns:
      keypoints: list of dicts or None
        {
          "x": float,
          "y": float,
          "visibility": float
        }
    """

    def __init__(self, min_visibility=0.2):
        self.min_visibility = min_visibility

        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    def process(self, frame):
        if frame is None:
            return []

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.pose.process(rgb)

        if not result.pose_landmarks:
            return []

        keypoints = []

        for lm in result.pose_landmarks.landmark:
            # Webcam-safe confidence gate
            if lm.visibility < self.min_visibility:
                keypoints.append(None)
            else:
                keypoints.append({
                    "x": float(lm.x),
                    "y": float(lm.y),
                    "visibility": float(lm.visibility),
                })

        return keypoints
