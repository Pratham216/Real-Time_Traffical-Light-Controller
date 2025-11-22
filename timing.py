"""
Adaptive Traffic Light Timing Calculation Module
Implements the adaptive timing algorithm based on vehicle counts
"""
from typing import Dict, List, Tuple
import numpy as np


class AdaptiveTimingController:
    """Calculates adaptive traffic light timings based on vehicle density"""
    
    def __init__(self, base_green: float = 5.0, scaling_factor: float = 1.0,
                 min_green: float = 5.0, max_green: float = 60.0,
                 yellow_time: float = 3.0, all_red_time: float = 1.0):
        """
        Initialize timing controller
        
        Args:
            base_green: Base green time in seconds (G0)
            scaling_factor: Scaling factor k (seconds per vehicle)
            min_green: Minimum green time in seconds (Gmin)
            max_green: Maximum green time in seconds (Gmax)
            yellow_time: Fixed yellow time in seconds
            all_red_time: Fixed all-red time in seconds
        """
        self.base_green = base_green
        self.scaling_factor = scaling_factor
        self.min_green = min_green
        self.max_green = max_green
        self.yellow_time = yellow_time
        self.all_red_time = all_red_time
        
        # Current signal state
        self.current_phase = {}  # lane_id -> 'green', 'yellow', 'red'
        self.phase_start_time = {}  # lane_id -> time when phase started
        
    def compute_timings(self, lane_counts: Dict[int, float], 
                       current_time: float = 0.0) -> Dict[int, Dict[str, float]]:
        """
        Compute adaptive green times for each lane
        
        Formula: Gi = min(max(Gmin, G0 + k * Ni), Gmax)
        
        Args:
            lane_counts: Dictionary mapping lane_id -> vehicle count (can be weighted or raw)
            current_time: Current simulation time (optional, for phase control)
            
        Returns:
            Dictionary mapping lane_id -> {
                'green': green_time,
                'yellow': yellow_time,
                'red': red_time (calculated),
                'total_cycle': total cycle time
            }
        """
        timings = {}
        
        # Calculate green time for each lane
        green_times = {}
        for lane_id, count in lane_counts.items():
            # Adaptive formula (works with both weighted and raw counts)
            green_time = self.base_green + self.scaling_factor * count
            # Apply constraints
            green_time = max(self.min_green, min(green_time, self.max_green))
            green_times[lane_id] = green_time
        
        # Calculate cycle time (sum of all green times + transitions)
        if green_times:
            max_green = max(green_times.values())
            total_cycle = max_green + self.yellow_time + self.all_red_time
        else:
            total_cycle = self.min_green + self.yellow_time + self.all_red_time
        
        # Assign timings
        for lane_id in lane_counts.keys():
            green_time = green_times.get(lane_id, self.min_green)
            red_time = total_cycle - green_time - self.yellow_time - self.all_red_time
            red_time = max(0, red_time)  # Ensure non-negative
            
            timings[lane_id] = {
                'green': round(green_time, 2),
                'yellow': round(self.yellow_time, 2),
                'red': round(red_time, 2),
                'total_cycle': round(total_cycle, 2),
                'vehicle_count': round(lane_counts.get(lane_id, 0), 2)  # Can be weighted or raw
            }
        
        return timings
    
    def get_current_phase(self, lane_id: int, current_time: float, 
                         timings: Dict[int, Dict[str, float]]) -> str:
        """
        Determine current signal phase for a lane based on time
        
        Args:
            lane_id: Lane identifier
            current_time: Current simulation time
            timings: Timings dictionary from compute_timings()
            
        Returns:
            Current phase: 'green', 'yellow', or 'red'
        """
        if lane_id not in timings:
            return 'red'
        
        timing = timings[lane_id]
        cycle_time = timing['total_cycle']
        
        # Simple round-robin: each lane gets its turn
        # For simplicity, we'll use a basic phase calculation
        # In a real system, this would be more sophisticated
        
        # Calculate phase within cycle
        phase_time = current_time % cycle_time
        
        if phase_time < timing['green']:
            return 'green'
        elif phase_time < timing['green'] + timing['yellow']:
            return 'yellow'
        else:
            return 'red'
    
    def update_phase_state(self, lane_id: int, phase: str, time: float):
        """Update the current phase state for a lane"""
        self.current_phase[lane_id] = phase
        self.phase_start_time[lane_id] = time

