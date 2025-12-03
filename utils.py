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
import os
from pathlib import Path


class TrafficLogger:
    """Logs traffic statistics over time"""
    
    def __init__(self):
        """Initialize logger with empty data structures"""
        self.log_data = {
            'timestamp': [],
            'frame_index': [],
            'latency_ms': [],  # Time between frames in milliseconds
            'throughput_vehicles_per_sec': [],  # Vehicles processed per second
            'total_vehicles': [],  # Total vehicles at each frame
        }
        self.frame_index = 0
        self.last_timestamp = None
        self.last_frame_time = None
        self.initial_vehicle_count = None
        self.final_vehicle_count = None
        
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
        import time as time_module
        
        current_time = self.frame_index / fps if fps > 0 else 0.0
        current_frame_time = time_module.time()
        
        # Calculate latency (time between frames)
        if self.last_frame_time is not None:
            latency_ms = (current_frame_time - self.last_frame_time) * 1000  # Convert to milliseconds
        else:
            latency_ms = 0.0
        
        # Calculate total vehicles
        total_vehicles = sum(raw_counts.values())
        
        # Store initial vehicle count
        if self.initial_vehicle_count is None:
            self.initial_vehicle_count = total_vehicles
        
        # Update final vehicle count
        self.final_vehicle_count = total_vehicles
        
        # Calculate throughput (vehicles per second)
        if self.last_timestamp is not None and current_time > self.last_timestamp:
            time_delta = current_time - self.last_timestamp
            if time_delta > 0:
                throughput = total_vehicles / time_delta
            else:
                throughput = 0.0
        else:
            throughput = 0.0
        
        self.log_data['timestamp'].append(current_time)
        self.log_data['frame_index'].append(self.frame_index)
        self.log_data['latency_ms'].append(latency_ms)
        self.log_data['throughput_vehicles_per_sec'].append(throughput)
        self.log_data['total_vehicles'].append(total_vehicles)
        
        self.last_timestamp = current_time
        self.last_frame_time = current_frame_time
        
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
    
    def get_summary_stats(self) -> Dict[str, float]:
        """Get summary statistics including initial/final counts and performance metrics"""
        if not self.log_data or len(self.log_data.get('timestamp', [])) == 0:
            return {}
        
        df = self.to_dataframe()
        if df.empty:
            return {}
        
        summary = {
            'initial_vehicle_count': self.initial_vehicle_count if self.initial_vehicle_count is not None else 0,
            'final_vehicle_count': self.final_vehicle_count if self.final_vehicle_count is not None else 0,
            'total_simulation_time_sec': df['timestamp'].max() if 'timestamp' in df.columns and len(df) > 0 else 0.0,
            'total_frames': len(df),
            'avg_latency_ms': df['latency_ms'].mean() if 'latency_ms' in df.columns else 0.0,
            'max_latency_ms': df['latency_ms'].max() if 'latency_ms' in df.columns else 0.0,
            'min_latency_ms': df['latency_ms'].min() if 'latency_ms' in df.columns else 0.0,
            'avg_throughput_vehicles_per_sec': df['throughput_vehicles_per_sec'].mean() if 'throughput_vehicles_per_sec' in df.columns else 0.0,
            'max_throughput_vehicles_per_sec': df['throughput_vehicles_per_sec'].max() if 'throughput_vehicles_per_sec' in df.columns else 0.0,
            'avg_total_vehicles': df['total_vehicles'].mean() if 'total_vehicles' in df.columns else 0.0,
            'max_total_vehicles': df['total_vehicles'].max() if 'total_vehicles' in df.columns else 0.0,
        }
        
        return summary
    
    def save_to_csv(self, filename: Optional[str] = None):
        """
        Save log data to CSV file with summary statistics
        
        Args:
            filename: Optional filename. If None, generates a timestamped filename.
            
        Returns:
            Full path to saved file
            
        Raises:
            ValueError: If no data to save
            IOError: If file cannot be written
        """
        # Check if there's any data to save
        if not self.log_data or len(self.log_data.get('timestamp', [])) == 0:
            raise ValueError("No statistics data to save. Please run simulation or process video first.")
        
        # Generate filename if not provided
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"traffic_stats_{timestamp}.csv"
        
        # Ensure filename has .csv extension
        if not filename.endswith('.csv'):
            filename += '.csv'
        
        # Get absolute path
        file_path = Path(filename).resolve()
        
        # Create directory if it doesn't exist
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert to DataFrame and save
        try:
            df = self.to_dataframe()
            if df.empty:
                raise ValueError("No statistics data to save. DataFrame is empty.")
            
            # Get summary statistics
            summary = self.get_summary_stats()
            
            # Create summary DataFrame
            summary_df = pd.DataFrame([summary])
            
            # Save main data
            df.to_csv(file_path, index=False)
            
            # Append summary statistics to a separate section in the same file
            # We'll add it as a comment section at the end
            summary_file = file_path.parent / f"{file_path.stem}_summary.csv"
            summary_df.to_csv(summary_file, index=False)
            
            return str(file_path)
        except Exception as e:
            raise IOError(f"Failed to save statistics file: {str(e)}")
    
    def reset(self):
        """Reset logger"""
        self.log_data = {
            'timestamp': [],
            'frame_index': [],
            'latency_ms': [],
            'throughput_vehicles_per_sec': [],
            'total_vehicles': [],
        }
        self.frame_index = 0
        self.last_timestamp = None
        self.last_frame_time = None
        self.initial_vehicle_count = None
        self.final_vehicle_count = None


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

