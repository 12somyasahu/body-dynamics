import cv2
import mediapipe as mp
import logging

logger = logging.getLogger(__name__)


class PoseEstimator:
    """
    MediaPipe Pose wrapper (webcam-safe, optimized for realtime).

    Returns:
      keypoints: list of dicts or None
        {
          "x": float,
          "y": float,
          "visibility": float
        }
    
    Performance improvements:
    - model_complexity=0 (lite model, 2-3x faster)
    - smooth_landmarks=False (you have Kalman filter)
    - Proper error handling
    """

    def __init__(self, min_visibility=0.2):
        self.min_visibility = min_visibility

        self.mp_pose = mp.solutions.pose
        
        # OPTIMIZATION: Use lite model + disable smoothing
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=0,        # ← CHANGED: 0=lite (2-3x faster than 1)
            smooth_landmarks=False,    # ← CHANGED: You already have Kalman!
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        
        logger.info("PoseEstimator initialized with lite model (complexity=0)")

    def process(self, frame):
        """
        Process a single frame and extract pose keypoints.
        
        Args:
            frame: BGR image from cv2 (numpy array)
            
        Returns:
            List of keypoint dicts, or empty list on failure
        """
        if frame is None:
            logger.warning("Received None frame")
            return []
        
        try:
            # Validate frame
            if frame.size == 0:
                logger.warning("Received empty frame")
                return []
            
            # Convert to RGB
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Run pose estimation
            result = self.pose.process(rgb)

            if not result.pose_landmarks:
                return []

            # Extract keypoints
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
        
        except cv2.error as e:
            logger.error(f"OpenCV error in pose estimation: {e}")
            return []
        
        except Exception as e:
            logger.error(f"Unexpected error in pose estimation: {e}", exc_info=True)
            return []
    
    def close(self):
        """Release MediaPipe resources."""
        try:
            self.pose.close()
            logger.info("PoseEstimator closed")
        except Exception as e:
            logger.error(f"Error closing PoseEstimator: {e}")
