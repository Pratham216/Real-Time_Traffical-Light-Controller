"""
Multi-Object Tracking Module
Handles vehicle tracking using ByteTrack via Ultralytics
"""
import cv2
import numpy as np
from ultralytics import YOLO
from typing import List, Dict, Tuple, Optional

# COCO class indices for vehicles
VEHICLE_CLASSES = [2, 3, 5, 7]


class VehicleTracker:
    """Wrapper for ByteTrack multi-object tracking"""
    
    def __init__(self, model_path: str = "yolov8n.pt", conf_threshold: float = 0.3,
                 device: str = "cuda", use_half: bool = True, tracker: str = "bytetrack.yaml",
                 vehicle_classes: Optional[List[int]] = None,
                 emergency_class_map: Optional[Dict[str, int]] = None):
        """
        Initialize YOLOv8 model with tracking
        
        Args:
            model_path: Path to YOLOv8 model weights (can be fine-tuned model)
            conf_threshold: Confidence threshold for detection
            device: Device to run inference on
            use_half: Whether to use FP16 half precision
            tracker: Tracker type (bytetrack.yaml or botsort.yaml)
            vehicle_classes: List of class IDs to detect (None = detect all)
            emergency_class_map: Dict mapping emergency types to class IDs
        """
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.device = device if device == "cuda" and self._check_cuda() else "cpu"
        self.use_half = use_half and self.device == "cuda"
        self.tracker = tracker
        
        # Vehicle classes to detect
        self.vehicle_classes = vehicle_classes if vehicle_classes else VEHICLE_CLASSES
        
        # Emergency vehicle class mapping (for fine-tuned models)
        self.emergency_class_map = emergency_class_map if emergency_class_map else {}
        
        # Get model class names (for fine-tuned models)
        try:
            self.model_names = self.model.names if hasattr(self.model, 'names') else {}
        except:
            self.model_names = {}
        
        # Optimize model
        if self.device == "cuda":
            self.model.to(self.device)
            if self.use_half:
                self.model.half = True
        self.model.fuse()
        
        # Track history for counting
        self.track_history: Dict[int, List[Tuple[int, int]]] = {}
        
    def _check_cuda(self) -> bool:
        """Check if CUDA is available"""
        try:
            import torch
            return torch.cuda.is_available()
        except:
            return False
    
    def track(self, frame: np.ndarray) -> List[Dict]:
        """
        Track vehicles in a frame
        
        Args:
            frame: Input frame (BGR format)
            
        Returns:
            List of tracked objects with keys: id, bbox (x1,y1,x2,y2), conf, class_id, center
        """
        # Run tracking with vehicle classes (include emergency if using fine-tuned model)
        classes_to_detect = self.vehicle_classes.copy()
        if self.emergency_class_map:
            classes_to_detect.extend([cid for cid in self.emergency_class_map.values() if cid is not None])
        
        results = self.model.track(
            frame,
            conf=self.conf_threshold,
            classes=classes_to_detect if classes_to_detect else None,  # None = detect all
            device=self.device,
            tracker=self.tracker,
            verbose=False,
            half=self.use_half,
            persist=True  # Maintain track IDs across frames
        )
        
        tracks = []
        if results and len(results) > 0:
            boxes = results[0].boxes
            if boxes is not None and len(boxes) > 0:
                # Check if tracking IDs are available
                track_ids = boxes.id
                if track_ids is not None:
                    track_ids = track_ids.cpu().numpy()
                else:
                    track_ids = [None] * len(boxes)
                
                for i, box in enumerate(boxes):
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = float(box.conf[0].cpu().numpy())
                    cls = int(box.cls[0].cpu().numpy())
                    track_id = int(track_ids[i]) if track_ids[i] is not None else None
                    
                    # Calculate center and bottom-center (for lane assignment)
                    center_x = int((x1 + x2) / 2)
                    center_y = int((y1 + y2) / 2)
                    bottom_y = int(y2)
                    
                    if track_id is not None:
                        # Update track history
                        if track_id not in self.track_history:
                            self.track_history[track_id] = []
                        self.track_history[track_id].append((center_x, bottom_y))
                        
                        # Keep only recent history (last 30 frames)
                        if len(self.track_history[track_id]) > 30:
                            self.track_history[track_id] = self.track_history[track_id][-30:]
                    
                    tracks.append({
                        'id': track_id,
                        'bbox': (int(x1), int(y1), int(x2), int(y2)),
                        'conf': conf,
                        'class_id': cls,
                        'center': (center_x, center_y),
                        'bottom_center': (center_x, bottom_y)
                    })
        
        # Clean up old tracks (not seen recently)
        self._cleanup_tracks(tracks)
        
        return tracks
    
    def _cleanup_tracks(self, current_tracks: List[Dict]):
        """Remove tracks that are no longer active"""
        active_ids = {track['id'] for track in current_tracks if track['id'] is not None}
        self.track_history = {tid: hist for tid, hist in self.track_history.items() 
                            if tid in active_ids}
    
    def get_active_track_ids(self) -> List[int]:
        """Get list of currently active track IDs"""
        return list(self.track_history.keys())
    
    def is_emergency_vehicle(self, class_id: int) -> Optional[str]:
        """
        Check if a class ID corresponds to an emergency vehicle
        
        Args:
            class_id: Detected class ID
            
        Returns:
            Emergency vehicle type ('ambulance', 'fire_truck', 'police') or None
        """
        for emergency_type, e_class_id in self.emergency_class_map.items():
            if e_class_id == class_id:
                return emergency_type
        return None

