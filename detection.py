"""
YOLOv8 Object Detection Module
Handles vehicle detection using Ultralytics YOLOv8
"""
import cv2
import numpy as np
from ultralytics import YOLO
from typing import List, Tuple, Optional, Dict

# COCO class indices for vehicles: car=2, motorcycle=3, bus=5, truck=7
# For fine-tuned models, emergency vehicle classes should be added:
# Example: ambulance=8, fire_truck=9, police=10 (depending on your custom model)
COCO_VEHICLE_CLASSES = [2, 3, 5, 7]  # Standard COCO classes

# Emergency vehicle class IDs (for fine-tuned models)
# Update these based on your custom model's class IDs
EMERGENCY_CLASSES = {
    'ambulance': None,  # Set to class ID from your fine-tuned model
    'fire_truck': None,
    'police': None
}


class VehicleDetector:
    """Wrapper for YOLOv8 vehicle detection"""
    
    def __init__(self, model_path: str = "yolov8n.pt", conf_threshold: float = 0.3, 
                 device: str = "cuda", use_half: bool = True, 
                 vehicle_classes: Optional[List[int]] = None,
                 emergency_class_map: Optional[Dict[str, int]] = None):
        """
        Initialize YOLOv8 model
        
        Args:
            model_path: Path to YOLOv8 model weights (can be fine-tuned model)
            conf_threshold: Confidence threshold for detection
            device: Device to run inference on ('cuda' or 'cpu')
            use_half: Whether to use FP16 half precision
            vehicle_classes: List of class IDs to detect (None = detect all)
            emergency_class_map: Dict mapping emergency types to class IDs
                                e.g., {'ambulance': 8, 'fire_truck': 9, 'police': 10}
        """
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.device = device if device == "cuda" and self._check_cuda() else "cpu"
        self.use_half = use_half and self.device == "cuda"
        
        # Vehicle classes to detect
        self.vehicle_classes = vehicle_classes if vehicle_classes else COCO_VEHICLE_CLASSES
        
        # Emergency vehicle class mapping (for fine-tuned models)
        self.emergency_class_map = emergency_class_map if emergency_class_map else {}
        
        # Optimize model
        if self.device == "cuda":
            self.model.to(self.device)
            if self.use_half:
                self.model.half = True
        self.model.fuse()
        
        # Get model class names (for fine-tuned models)
        try:
            self.model_names = self.model.names if hasattr(self.model, 'names') else {}
        except:
            self.model_names = {}
        
    def _check_cuda(self) -> bool:
        """Check if CUDA is available"""
        try:
            import torch
            return torch.cuda.is_available()
        except:
            return False
    
    def detect(self, frame: np.ndarray) -> List[Tuple[int, int, int, int, float, int]]:
        """
        Detect vehicles in a frame
        
        Args:
            frame: Input frame (BGR format)
            
        Returns:
            List of detections: (x1, y1, x2, y2, conf, class_id)
        """
        # Run detection with vehicle classes only
        # If using fine-tuned model, include emergency vehicle classes
        classes_to_detect = self.vehicle_classes.copy()
        if self.emergency_class_map:
            classes_to_detect.extend([cid for cid in self.emergency_class_map.values() if cid is not None])
        
        results = self.model.predict(
            frame,
            conf=self.conf_threshold,
            classes=classes_to_detect if classes_to_detect else None,  # None = detect all
            device=self.device,
            verbose=False,
            half=self.use_half
        )
        
        detections = []
        if results and len(results) > 0:
            boxes = results[0].boxes
            if boxes is not None and len(boxes) > 0:
                for box in boxes:
                    # Get bounding box coordinates
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = float(box.conf[0].cpu().numpy())
                    cls = int(box.cls[0].cpu().numpy())
                    
                    detections.append((
                        int(x1), int(y1), int(x2), int(y2),
                        conf, cls
                    ))
        
        return detections
    
    def get_model_info(self) -> dict:
        """Get model and device information"""
        return {
            "device": self.device,
            "half_precision": self.use_half,
            "conf_threshold": self.conf_threshold,
            "vehicle_classes": self.vehicle_classes,
            "emergency_classes": self.emergency_class_map,
            "model_names": self.model_names
        }
    
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

