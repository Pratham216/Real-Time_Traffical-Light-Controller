"""
Dynamic Lane Detection Module
Detects lanes using Hough Transform and assigns vehicles to lanes
"""
import cv2
import numpy as np
from typing import List, Tuple, Dict, Optional
from scipy import stats
from sklearn.cluster import KMeans


class LaneDetector:
    """Detects lanes and assigns vehicles to lane regions"""
    
    def __init__(self, num_lanes: int = 2, roi_ratio: float = 0.6):
        """
        Initialize lane detector
        
        Args:
            num_lanes: Expected number of lanes (can be auto-detected)
            roi_ratio: Ratio of frame height to use for ROI (0-1)
        """
        self.num_lanes = num_lanes
        self.roi_ratio = roi_ratio
        self.lane_boundaries: List[Tuple[int, int]] = []  # List of (x_left, x_right) for each lane
        self.lane_lines: List[Tuple[int, int, int, int]] = []  # List of line segments
        self.frame_shape: Optional[Tuple[int, int]] = None
        self.is_initialized = False
        
    def detect_lanes(self, frame: np.ndarray, auto_detect_lanes: bool = True) -> bool:
        """
        Detect lanes in a frame using Hough Transform
        
        Args:
            frame: Input frame (BGR format)
            auto_detect_lanes: Whether to auto-detect number of lanes
            
        Returns:
            True if lanes were detected successfully
        """
        self.frame_shape = frame.shape[:2]  # (height, width)
        height, width = self.frame_shape
        
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Define ROI (Region of Interest) - lower portion of frame
        roi_y_start = int(height * (1 - self.roi_ratio))
        roi = gray[roi_y_start:, :]
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(roi, (5, 5), 0)
        
        # Canny edge detection
        edges = cv2.Canny(blurred, 50, 150, apertureSize=3)
        
        # Hough Line Transform
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi/180,
            threshold=50,
            minLineLength=50,
            maxLineGap=20
        )
        
        if lines is None or len(lines) == 0:
            return False
        
        # Filter and cluster lines
        vertical_lines = self._filter_vertical_lines(lines, width)
        
        if len(vertical_lines) < 2:
            # Fallback: use simple x-coordinate partitioning
            return self._simple_partition(width)
        
        # Cluster lines to find lane boundaries
        lane_x_positions = self._cluster_lanes(vertical_lines, auto_detect_lanes)
        
        if len(lane_x_positions) < 2:
            return self._simple_partition(width)
        
        # Create lane boundaries
        self.lane_boundaries = []
        lane_x_positions = sorted(lane_x_positions)
        
        # Add left boundary (frame edge)
        self.lane_boundaries.append((0, lane_x_positions[0]))
        
        # Add middle lanes
        for i in range(len(lane_x_positions) - 1):
            self.lane_boundaries.append((lane_x_positions[i], lane_x_positions[i+1]))
        
        # Add right boundary (frame edge)
        self.lane_boundaries.append((lane_x_positions[-1], width))
        
        self.num_lanes = len(self.lane_boundaries)
        self.is_initialized = True
        
        return True
    
    def _filter_vertical_lines(self, lines: np.ndarray, width: int, 
                              angle_threshold: float = 20.0) -> List[Tuple[int, int, int, int]]:
        """Filter lines that are approximately vertical"""
        vertical_lines = []
        
        for line in lines:
            x1, y1, x2, y2 = line[0]
            
            # Calculate angle
            if x2 == x1:
                angle = 90
            else:
                angle = np.abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
            
            # Filter vertical lines (angle close to 90 degrees)
            if abs(angle - 90) < angle_threshold or abs(angle - 90) > (180 - angle_threshold):
                vertical_lines.append((x1, y1, x2, y2))
        
        return vertical_lines
    
    def _cluster_lanes(self, lines: List[Tuple[int, int, int, int]], 
                      auto_detect: bool = True) -> List[int]:
        """Cluster vertical lines to find lane boundaries"""
        # Extract x-coordinates of line centers
        x_coords = []
        for x1, y1, x2, y2 in lines:
            x_center = (x1 + x2) // 2
            x_coords.append(x_center)
        
        if len(x_coords) < 2:
            return []
        
        x_coords = np.array(x_coords).reshape(-1, 1)
        
        # Determine optimal number of clusters
        n_clusters = self.num_lanes - 1 if not auto_detect else self._find_optimal_clusters(x_coords)
        n_clusters = min(n_clusters, len(x_coords))
        
        if n_clusters < 1:
            n_clusters = 1
        
        # K-means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        kmeans.fit(x_coords)
        
        # Get cluster centers (lane boundary x-positions)
        lane_x_positions = sorted([int(center[0]) for center in kmeans.cluster_centers_])
        
        return lane_x_positions
    
    def _find_optimal_clusters(self, x_coords: np.ndarray, max_clusters: int = 5) -> int:
        """Find optimal number of clusters using elbow method"""
        if len(x_coords) <= 2:
            return 1
        
        max_clusters = min(max_clusters, len(x_coords))
        inertias = []
        
        for k in range(1, max_clusters + 1):
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(x_coords)
            inertias.append(kmeans.inertia_)
        
        # Simple elbow detection (simplified)
        if len(inertias) >= 3:
            # Find largest drop in inertia
            diffs = [inertias[i] - inertias[i+1] for i in range(len(inertias)-1)]
            if diffs:
                optimal = np.argmax(diffs) + 1
                return max(1, optimal)
        
        return max(1, max_clusters // 2)
    
    def _simple_partition(self, width: int) -> bool:
        """Fallback: simple equal-width partitioning"""
        if self.num_lanes <= 0:
            self.num_lanes = 2  # Default to 2 lanes
        
        lane_width = width // self.num_lanes
        self.lane_boundaries = []
        
        for i in range(self.num_lanes):
            x_left = i * lane_width
            x_right = (i + 1) * lane_width if i < self.num_lanes - 1 else width
            self.lane_boundaries.append((x_left, x_right))
        
        self.is_initialized = True
        return True
    
    def assign_lane(self, x: int, y: int) -> Optional[int]:
        """
        Assign a point (x, y) to a lane based on x-coordinate
        
        Args:
            x: X coordinate (usually bottom-center x of vehicle)
            y: Y coordinate (usually bottom-center y of vehicle)
            
        Returns:
            Lane ID (0-indexed) or None if outside all lanes
        """
        if not self.is_initialized or len(self.lane_boundaries) == 0:
            return None
        
        for lane_id, (x_left, x_right) in enumerate(self.lane_boundaries):
            if x_left <= x < x_right:
                return lane_id
        
        # Handle edge case: if x equals rightmost boundary
        if x == self.lane_boundaries[-1][1]:
            return len(self.lane_boundaries) - 1
        
        return None
    
    def get_lane_boundaries(self) -> List[Tuple[int, int]]:
        """Get current lane boundaries"""
        return self.lane_boundaries
    
    def visualize_lanes(self, frame: np.ndarray) -> np.ndarray:
        """Draw lane boundaries on frame for visualization"""
        if not self.is_initialized:
            return frame
        
        vis_frame = frame.copy()
        height, width = frame.shape[:2]
        
        for x_left, x_right in self.lane_boundaries:
            # Draw vertical lines for lane boundaries
            cv2.line(vis_frame, (x_left, 0), (x_left, height), (0, 255, 255), 2)
            cv2.line(vis_frame, (x_right, 0), (x_right, height), (0, 255, 255), 2)
        
        return vis_frame

