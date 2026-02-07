import cv2
import mediapipe as mp
#another one

class PoseEstimator:
    """
    MediaPipe Pose wrapper with confidence-aware landmarks.

    Returns:
      keypoints: list of dicts or None
        {
          "x": float,
          "y": float,
          "visibility": float,
          "presence": float
        }
    """

    def __init__(
        self,
        min_visibility=0.5,
        min_presence=0.5
    ):
        self.min_visibility = min_visibility
        self.min_presence = min_presence

        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    def process(self, frame):
        """
        frame: BGR image (OpenCV)
        returns: list of landmark dicts or None
        """

        if frame is None:
            return []

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.pose.process(rgb)

        keypoints = []

        if not result.pose_landmarks:
            return []

        for lm in result.pose_landmarks.landmark:
            # Hard confidence gate
            if lm.visibility < self.min_visibility or lm.presence < self.min_presence:
                keypoints.append(None)
            else:
                keypoints.append({
                    "x": float(lm.x),
                    "y": float(lm.y),
                    "visibility": float(lm.visibility),
                    "presence": float(lm.presence),
                })

        return keypoints
