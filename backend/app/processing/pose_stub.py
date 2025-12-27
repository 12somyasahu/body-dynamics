import cv2
import numpy as np
import mediapipe as mp
from typing import Dict, List, Tuple


class PoseEstimatorStub:
    """
    MediaPipe Pose-based estimator.
    Drop-in replacement for the stub.
    """

    def __init__(self):
        self.model_name = "mediapipe_pose"

        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    def estimate(self, frame: bytes) -> Dict[str, List[Tuple[float, float]]]:
        """
        Input: JPEG bytes
        Output: normalized pose landmarks
        """

        # Decode JPEG → numpy image
        np_arr = np.frombuffer(frame, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if img is None:
            return {}

        # BGR → RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Run MediaPipe Pose
        result = self.pose.process(img_rgb)

        if not result.pose_landmarks:
            return {}

        keypoints = []
        for lm in result.pose_landmarks.landmark:
            keypoints.append((lm.x, lm.y))  # normalized [0,1]

        return {
            "person_0": keypoints
        }
