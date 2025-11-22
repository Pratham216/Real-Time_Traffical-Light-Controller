"""
Emergency Vehicle Detection Module
Detects emergency vehicles (ambulance, fire truck, police car) using visual features
"""
import cv2
import numpy as np
from typing import Dict, Tuple, Optional, List


class EmergencyVehicleDetector:
    """Detects emergency vehicles based on color and visual features"""
    
    def __init__(self, enable_detection: bool = True):
        """
        Initialize emergency vehicle detector
        
        Args:
            enable_detection: Whether to enable emergency vehicle detection
        """
        self.enable_detection = enable_detection
        
        # Color ranges for emergency vehicles (HSV color space)
        # Red color range (ambulances, fire trucks)
        self.red_lower1 = np.array([0, 50, 50])
        self.red_upper1 = np.array([10, 255, 255])
        self.red_lower2 = np.array([170, 50, 50])
        self.red_upper2 = np.array([180, 255, 255])
        
        # Blue color range (police cars, ambulances)
        self.blue_lower = np.array([100, 50, 50])
        self.blue_upper = np.array([130, 255, 255])
        
        # Yellow/Amber color range (emergency vehicles)
        self.yellow_lower = np.array([20, 50, 50])
        self.yellow_upper = np.array([30, 255, 255])
        
        # White color range (ambulances, police cars)
        self.white_lower = np.array([0, 0, 200])
        self.white_upper = np.array([180, 30, 255])
    
    def detect_emergency_vehicle(self, frame: np.ndarray, bbox: Tuple[int, int, int, int], 
                                 class_id: int) -> Optional[str]:
        """
        Detect if a vehicle is an emergency vehicle based on visual features
        
        Args:
            frame: Input frame (BGR format)
            bbox: Bounding box (x1, y1, x2, y2)
            class_id: COCO class ID (2=car, 7=truck, etc.)
            
        Returns:
            Emergency vehicle type: 'ambulance', 'fire_truck', 'police', or None
        """
        if not self.enable_detection:
            return None
        
        x1, y1, x2, y2 = bbox
        # Extract vehicle region (expand slightly to capture lights on top/sides)
        h, w = frame.shape[:2]
        x1_expanded = max(0, x1 - 5)
        y1_expanded = max(0, y1 - 10)  # Check top for roof lights
        x2_expanded = min(w, x2 + 5)
        y2_expanded = min(h, y2 + 5)
        
        vehicle_roi = frame[y1_expanded:y2_expanded, x1_expanded:x2_expanded]
        
        if vehicle_roi.size == 0:
            return None
        
        # Convert to HSV for better color detection
        hsv = cv2.cvtColor(vehicle_roi, cv2.COLOR_BGR2HSV)
        
        # Detect red lights (ambulance, fire truck)
        red_mask1 = cv2.inRange(hsv, self.red_lower1, self.red_upper1)
        red_mask2 = cv2.inRange(hsv, self.red_lower2, self.red_upper2)
        red_mask = cv2.bitwise_or(red_mask1, red_mask2)
        red_pixels = cv2.countNonZero(red_mask)
        
        # Detect blue lights (police, ambulance)
        blue_mask = cv2.inRange(hsv, self.blue_lower, self.blue_upper)
        blue_pixels = cv2.countNonZero(blue_mask)
        
        # Detect yellow/amber lights (emergency vehicles)
        yellow_mask = cv2.inRange(hsv, self.yellow_lower, self.yellow_upper)
        yellow_pixels = cv2.countNonZero(yellow_mask)
        
        # Detect white (ambulances often have white sections)
        white_mask = cv2.inRange(hsv, self.white_lower, self.white_upper)
        white_pixels = cv2.countNonZero(white_mask)
        
        # Calculate percentage of pixels with emergency colors
        total_pixels = vehicle_roi.shape[0] * vehicle_roi.shape[1]
        red_ratio = red_pixels / total_pixels if total_pixels > 0 else 0
        blue_ratio = blue_pixels / total_pixels if total_pixels > 0 else 0
        yellow_ratio = yellow_pixels / total_pixels if total_pixels > 0 else 0
        white_ratio = white_pixels / total_pixels if total_pixels > 0 else 0
        
        # Detection thresholds (can be adjusted)
        emergency_threshold = 0.02  # 2% of vehicle area has emergency colors
        high_emergency_threshold = 0.05  # 5% for strong detection
        
        # Check for ambulance (red + blue, often with white)
        if (red_ratio > emergency_threshold and blue_ratio > emergency_threshold) or \
           (red_ratio > high_emergency_threshold and white_ratio > 0.15):
            return 'ambulance'
        
        # Check for fire truck (primarily red, often larger/truck class)
        if red_ratio > high_emergency_threshold and class_id == 7:  # truck
            return 'fire_truck'
        
        # Check for police car (blue lights, often car class)
        if blue_ratio > emergency_threshold and class_id == 2:  # car
            return 'police'
        
        # Check for general emergency vehicle (any significant emergency colors)
        if (red_ratio > high_emergency_threshold or blue_ratio > high_emergency_threshold) and \
           (yellow_ratio > emergency_threshold or red_ratio > emergency_threshold):
            # If it's a truck, likely fire truck
            if class_id == 7:
                return 'fire_truck'
            # Otherwise likely ambulance
            return 'ambulance'
        
        return None
    
    def is_emergency_vehicle(self, frame: np.ndarray, bbox: Tuple[int, int, int, int], 
                            class_id: int) -> bool:
        """
        Check if vehicle is an emergency vehicle (simplified boolean check)
        
        Args:
            frame: Input frame
            bbox: Bounding box
            class_id: COCO class ID
            
        Returns:
            True if emergency vehicle detected
        """
        return self.detect_emergency_vehicle(frame, bbox, class_id) is not None








