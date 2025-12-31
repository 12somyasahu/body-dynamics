import cv2
import mediapipe as mp


class PoseEstimator:
    """
    Pose estimator using MediaPipe.
    Returns normalized keypoints: [ [x, y], ... ]
    """

    def __init__(self):
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
        returns: list of [x, y] normalized coordinates
        """

        if frame is None:
            return []

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.pose.process(rgb)

        keypoints = []

        if result.pose_landmarks:
            for lm in result.pose_landmarks.landmark:
                keypoints.append([lm.x, lm.y])

        return keypoints
