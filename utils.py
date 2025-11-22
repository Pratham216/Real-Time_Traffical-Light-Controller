"""
Utility Functions Module
Helper functions for logging, data structures, and visualization
"""
import pandas as pd
import numpy as np
import cv2
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import time


class TrafficLogger:
    """Logs traffic statistics over time"""
    
    def __init__(self):
        """Initialize logger with empty data structures"""
        self.log_data = {
            'timestamp': [],
            'frame_index': [],
        }
        self.frame_index = 0
        
    def add_lane_columns(self, lane_ids: List[int], include_weighted: bool = False):
        """Add columns for new lanes"""
        for lane_id in lane_ids:
            self.log_data[f'lane_{lane_id}_count'] = []
            if include_weighted:
                self.log_data[f'lane_{lane_id}_weighted_count'] = []
            self.log_data[f'lane_{lane_id}_green'] = []
            self.log_data[f'lane_{lane_id}_yellow'] = []
            self.log_data[f'lane_{lane_id}_red'] = []
    
    def log_frame(self, raw_counts: Dict[int, int], 
                  timings: Dict[int, Dict[str, float]], 
                  fps: float = 30.0):
        """
        Log statistics for current frame
        
        Args:
            raw_counts: Dictionary mapping lane_id -> raw vehicle count
            timings: Dictionary mapping lane_id -> timing info (may include weighted_count)
            fps: Video frame rate
        """
        current_time = self.frame_index / fps if fps > 0 else 0.0
        
        self.log_data['timestamp'].append(current_time)
        self.log_data['frame_index'].append(self.frame_index)
        
        # Get all unique lane IDs from both raw_counts and timings
        all_lane_ids = set(raw_counts.keys()) | set(timings.keys())
        
        # Ensure all lanes are initialized (add missing lanes on the fly)
        missing_lanes = [lid for lid in all_lane_ids 
                        if f'lane_{lid}_count' not in self.log_data]
        if missing_lanes:
            # Check if weighted columns exist in log_data or if timings contain weighted_count
            include_weighted = (any('weighted_count' in key for key in self.log_data.keys()) or
                              any('weighted_count' in timing for timing in timings.values()))
            self.add_lane_columns(missing_lanes, include_weighted=include_weighted)
        
        # Log per-lane data
        for lane_id in sorted(all_lane_ids):
            raw_count = raw_counts.get(lane_id, 0)
            timing = timings.get(lane_id, {})
            weighted_count = timing.get('weighted_count', raw_count)
            
            self.log_data[f'lane_{lane_id}_count'].append(raw_count)
            # Log weighted count if column exists
            if f'lane_{lane_id}_weighted_count' in self.log_data:
                self.log_data[f'lane_{lane_id}_weighted_count'].append(weighted_count)
            self.log_data[f'lane_{lane_id}_green'].append(timing.get('green', 0))
            self.log_data[f'lane_{lane_id}_yellow'].append(timing.get('yellow', 0))
            self.log_data[f'lane_{lane_id}_red'].append(timing.get('red', 0))
        
        self.frame_index += 1
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert log data to pandas DataFrame"""
        return pd.DataFrame(self.log_data)
    
    def save_to_csv(self, filename: str = "traffic_stats.csv"):
        """Save log data to CSV file"""
        df = self.to_dataframe()
        df.to_csv(filename, index=False)
        return filename
    
    def reset(self):
        """Reset logger"""
        self.log_data = {
            'timestamp': [],
            'frame_index': [],
        }
        self.frame_index = 0


def draw_vehicle_boxes(frame: np.ndarray, tracks: List[Dict], 
                      draw_ids: bool = False, emergency_info: Optional[Dict[int, str]] = None) -> np.ndarray:
    """
    Draw bounding boxes on frame for tracked vehicles
    
    Args:
        frame: Input frame
        tracks: List of tracked vehicles with 'bbox' key
        draw_ids: Whether to draw track IDs (default False per requirements)
        emergency_info: Optional dictionary mapping track_id -> emergency_type
        
    Returns:
        Annotated frame
    """
    annotated = frame.copy()
    
    for track in tracks:
        if track['bbox'] is not None:
            x1, y1, x2, y2 = track['bbox']
            track_id = track.get('id')
            
            # Use green color for all vehicles (no differentiation)
            color = (0, 255, 0)  # Green for all vehicles
            thickness = 2
            
            # Draw rectangle
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)
            
            # Optionally draw ID (disabled by default)
            if draw_ids and track_id is not None:
                cv2.putText(annotated, f"ID:{track_id}", (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    return annotated


def count_vehicles_per_lane(tracks: List[Dict], lane_detector) -> Dict[int, List[Dict]]:
    """
    Count vehicles per lane with their full information
    
    Args:
        tracks: List of tracked vehicles with 'id', 'class_id', 'bottom_center'
        lane_detector: LaneDetector instance
        
    Returns:
        Dictionary mapping lane_id -> list of vehicle dictionaries with id and class_id
    """
    lane_vehicles: Dict[int, List[Dict]] = {}
    
    for track in tracks:
        if track['id'] is not None and track['bottom_center'] is not None:
            x, y = track['bottom_center']
            lane_id = lane_detector.assign_lane(x, y)
            
            if lane_id is not None:
                if lane_id not in lane_vehicles:
                    lane_vehicles[lane_id] = []
                # Store vehicle info including class_id for weighting
                lane_vehicles[lane_id].append({
                    'id': track['id'],
                    'class_id': track.get('class_id', 2)  # Default to car if missing
                })
    
    return lane_vehicles


def get_lane_counts(lane_vehicles: Dict[int, List[Dict]]) -> Dict[int, int]:
    """
    Get unique vehicle counts per lane (raw count)
    
    Args:
        lane_vehicles: Dictionary mapping lane_id -> list of vehicle dicts
        
    Returns:
        Dictionary mapping lane_id -> unique count
    """
    return {lane_id: len(set(veh['id'] for veh in vehicles)) 
            for lane_id, vehicles in lane_vehicles.items()}


def get_weighted_lane_counts(lane_vehicles: Dict[int, List[Dict]], 
                             weight_manager, emergency_info: Optional[Dict[int, str]] = None) -> Dict[int, float]:
    """
    Get weighted vehicle counts per lane
    
    Args:
        lane_vehicles: Dictionary mapping lane_id -> list of vehicle dicts with class_id
        weight_manager: VehicleWeightManager instance
        emergency_info: Optional dictionary mapping track_id -> emergency_type
        
    Returns:
        Dictionary mapping lane_id -> weighted count
    """
    weighted_counts = {}
    
    for lane_id, vehicles in lane_vehicles.items():
        # Get unique vehicles (by ID) to avoid double counting
        unique_vehicles = {}
        for veh in vehicles:
            veh_id = veh['id']
            if veh_id not in unique_vehicles:
                unique_vehicles[veh_id] = veh
        
        # Calculate weighted sum
        total_weight = 0.0
        for veh in unique_vehicles.values():
            class_id = veh.get('class_id', 2)  # Default to car
            veh_id = veh.get('id')
            emergency_type = emergency_info.get(veh_id) if emergency_info else None
            total_weight += weight_manager.get_weight(class_id, emergency_type)
        
        weighted_counts[lane_id] = total_weight
    
    return weighted_counts


def resize_frame(frame: np.ndarray, target_width: int = 640, 
                target_height: int = 480) -> np.ndarray:
    """Resize frame to target dimensions"""
    return cv2.resize(frame, (target_width, target_height))


def format_timing_table(timings: Dict[int, Dict[str, float]]) -> pd.DataFrame:
    """
    Format timings dictionary as a pandas DataFrame for display
    
    Args:
        timings: Dictionary from timing controller
        
    Returns:
        Formatted DataFrame
    """
    rows = []
    for lane_id in sorted(timings.keys()):
        timing = timings[lane_id]
        raw_count = timing.get('raw_count', timing.get('vehicle_count', 0))
        weighted_count = timing.get('weighted_count', raw_count)
        
        # Always use string to avoid PyArrow type errors
        # Show both raw and weighted counts if weighted is different
        if 'weighted_count' in timing and abs(weighted_count - raw_count) > 0.1:
            vehicle_display = f"{int(raw_count)} ({weighted_count:.1f})"
        else:
            vehicle_display = str(int(raw_count))
        
        # Map lane IDs to directions for 4-way intersection
        direction_names = {0: "North", 1: "East", 2: "South", 3: "West"}
        lane_name = direction_names.get(lane_id, f"Lane {lane_id + 1}")
        
        rows.append({
            'Lane': lane_name,
            'Vehicles': vehicle_display,  # Always string
            'Green (s)': timing.get('green', 0),
            'Yellow (s)': timing.get('yellow', 0),
            'Red (s)': timing.get('red', 0),
            'Cycle Time (s)': timing.get('total_cycle', 0)
        })
    
    return pd.DataFrame(rows)

