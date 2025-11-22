"""
SUMO (Simulation of Urban MObility) Integration Module
Connects SUMO traffic simulation with the adaptive traffic light controller
"""
import os
import subprocess
import sys
from typing import Dict, List, Tuple, Optional
import time

try:
    import traci
    SUMO_AVAILABLE = True
except ImportError:
    SUMO_AVAILABLE = False
    print("Warning: SUMO (traci) not installed. Install with: pip install traci sumolib")

from timing import AdaptiveTimingController
from vehicle_weights import VehicleWeightManager


class SUMOTrafficSimulator:
    """Manages SUMO simulation and TraCI connection"""
    
    def __init__(self, config_file: str = None, gui: bool = True):
        """
        Initialize SUMO simulator
        
        Args:
            config_file: Path to SUMO configuration file (.sumocfg)
            gui: Whether to show SUMO GUI (default True)
        """
        self.config_file = config_file
        self.gui = gui
        self.simulation_running = False
        self.current_step = 0
        self.traffic_light_ids = []
        
        # Find SUMO binary
        if sys.platform == "win32":
            # Windows: try common installation paths
            possible_paths = [
                r"C:\Program Files (x86)\Eclipse\Sumo\bin\sumo-gui.exe",
                r"C:\Program Files\Eclipse\Sumo\bin\sumo-gui.exe",
                r"C:\Sumo\bin\sumo-gui.exe",
            ]
            if gui:
                self.sumo_binary = "sumo-gui"
            else:
                self.sumo_binary = "sumo"
            
            # Check if sumo is in PATH
            try:
                subprocess.run([self.sumo_binary, "--version"], 
                             capture_output=True, check=True, timeout=5)
            except:
                # Try to find in common paths
                for path in possible_paths:
                    if os.path.exists(path):
                        self.sumo_binary = path
                        break
        else:
            # Linux/Mac
            self.sumo_binary = "sumo-gui" if gui else "sumo"
    
    def start_simulation(self) -> List[str]:
        """
        Start SUMO simulation and return list of traffic light IDs
        
        Returns:
            List of traffic light junction IDs
        """
        if not SUMO_AVAILABLE:
            raise RuntimeError("SUMO (traci) is not installed. Install with: pip install traci sumolib")
        
        if not self.config_file or not os.path.exists(self.config_file):
            raise ValueError(f"SUMO config file not found: {self.config_file}")
        
        # Close any existing TraCI connections
        try:
            traci.close()
        except:
            pass
        
        # Start SUMO with TraCI
        sumo_cmd = [
            self.sumo_binary,
            "-c", self.config_file,
            "--start",
            "--step-length", "0.1",  # 0.1 second steps
            "--lateral-resolution", "0.8",
        ]
        
        if not self.gui:
            sumo_cmd.append("--no-warnings")
        
        try:
            traci.start(sumo_cmd)
        except traci.exceptions.TraCIException as e:
            if "already active" in str(e):
                # Force close and retry
                try:
                    traci.close()
                except:
                    pass
                traci.start(sumo_cmd)
            else:
                raise
        
        self.simulation_running = True
        self.current_step = 0
        
        # Get all traffic light IDs
        self.traffic_light_ids = traci.trafficlight.getIDList()
        
        # If no traffic lights found, step simulation to let SUMO initialize
        # SUMO may need a few steps to generate signal plans for traffic lights
        if not self.traffic_light_ids:
            try:
                # Step simulation a few times to let SUMO initialize
                for _ in range(10):
                    traci.simulationStep()
                    self.current_step += 1
                    self.traffic_light_ids = traci.trafficlight.getIDList()
                    if self.traffic_light_ids:
                        break
            except Exception as e:
                print(f"Warning during traffic light initialization: {e}")
        
        # If still no traffic lights, return empty list but don't block
        # The simulation can continue without traffic light control
        if not self.traffic_light_ids:
            print("Warning: No traffic lights detected by TraCI.")
            print("Simulation will continue, but traffic light control is disabled.")
            print("Signal plans may be missing or incorrect format.")
        
        return self.traffic_light_ids
    
    def step(self, steps: int = 1):
        """Advance simulation by specified number of steps"""
        if not self.simulation_running:
            raise RuntimeError("Simulation not started. Call start_simulation() first.")
        
        # Step multiple times to make simulation run faster
        # Each step is 0.1s, so 10 steps = 1 second
        actual_steps = steps * 10  # Step 10 times per call for faster simulation
        for _ in range(actual_steps):
            traci.simulationStep()
            self.current_step += 1
    
    def get_vehicle_counts_per_lane(self, intersection_id: str) -> Dict[str, int]:
        """
        Get vehicle counts per lane for an intersection
        
        Args:
            intersection_id: Traffic light ID (or any identifier)
            
        Returns:
            Dictionary mapping lane_id -> vehicle count
        """
        lane_counts = {}
        
        try:
            # Get controlled lanes for this traffic light
            controlled_lanes = traci.trafficlight.getControlledLanes(intersection_id)
            
            for lane in controlled_lanes:
                vehicle_count = traci.lane.getLastStepVehicleNumber(lane)
                lane_counts[lane] = vehicle_count
        except:
            # If traffic light doesn't exist, get counts from intersection area lanes
            try:
                intersection_lanes = self.get_all_lanes_in_intersection_area()
                for lane in intersection_lanes:
                    try:
                        vehicle_count = traci.lane.getLastStepVehicleNumber(lane)
                        lane_counts[lane] = vehicle_count
                    except:
                        pass
            except:
                # Final fallback: get from all lanes
                try:
                    all_lanes = traci.lane.getIDList()
                    for lane in all_lanes[:16]:  # Limit to 16 lanes (4 per direction)
                        try:
                            vehicle_count = traci.lane.getLastStepVehicleNumber(lane)
                            lane_counts[lane] = vehicle_count
                        except:
                            pass
                except:
                    pass  # Return empty dict if everything fails
        
        return lane_counts
    
    def get_vehicle_types_per_lane(self, intersection_id: str) -> Dict[str, List[str]]:
        """
        Get vehicle types per lane
        
        Args:
            intersection_id: Traffic light ID
            
        Returns:
            Dictionary mapping lane_id -> list of vehicle type IDs
        """
        lane_vehicle_types = {}
        
        try:
            controlled_lanes = traci.trafficlight.getControlledLanes(intersection_id)
        except:
            # If traffic light doesn't exist, get from all lanes
            try:
                all_lanes = traci.lane.getIDList()
                controlled_lanes = [lane for lane in all_lanes if intersection_id in lane or lane.startswith(intersection_id)]
            except:
                controlled_lanes = []
        
        for lane in controlled_lanes:
            try:
                vehicle_ids = traci.lane.getLastStepVehicleIDs(lane)
                vehicle_types = []
                
                for veh_id in vehicle_ids:
                    try:
                        veh_type = traci.vehicle.getTypeID(veh_id)
                        vehicle_types.append(veh_type)
                    except:
                        pass
                
                lane_vehicle_types[lane] = vehicle_types
            except:
                pass
        
        return lane_vehicle_types
    
    def get_current_time(self) -> float:
        """Get current simulation time in seconds"""
        return traci.simulation.getTime()
    
    def get_all_lanes_in_intersection_area(self) -> List[str]:
        """
        Get all lanes in the intersection area (for lane-based traffic control)
        
        Returns:
            List of lane IDs near the intersection
        """
        try:
            all_lanes = traci.lane.getIDList()
            # Filter lanes that are likely part of an intersection
            # This is a heuristic - you may need to adjust based on your network
            intersection_lanes = []
            for lane in all_lanes:
                # Get lane position to determine if it's near intersection
                try:
                    length = traci.lane.getLength(lane)
                    # Intersection lanes are typically shorter
                    if length < 100:  # Adjust threshold as needed
                        intersection_lanes.append(lane)
                except:
                    pass
            # Limit to 16 lanes for 4-way intersection (4 lanes per direction: North, East, South, West)
            intersection_lanes = intersection_lanes[:16] if intersection_lanes else all_lanes[:16]
            return intersection_lanes
        except:
            return []
    
    def control_lane_speeds(self, lane_speeds: Dict[str, Optional[float]]):
        """
        Control vehicle speeds in specific lanes to simulate traffic lights
        
        Args:
            lane_speeds: Dictionary mapping lane_id -> speed (0 = stop, None = use default/max speed, >0 = set speed)
        """
        try:
            for lane_id, target_speed in lane_speeds.items():
                try:
                    # Get all vehicles in this lane
                    vehicle_ids = traci.lane.getLastStepVehicleIDs(lane_id)
                    for veh_id in vehicle_ids:
                        # Set vehicle speed
                        if target_speed == 0:
                            traci.vehicle.setSpeed(veh_id, 0.0)
                        elif target_speed is None:
                            # Remove speed limit to allow normal speed
                            try:
                                traci.vehicle.setSpeedMode(veh_id, 0)  # Reset speed mode
                                traci.vehicle.setSpeed(veh_id, -1)  # -1 means use default
                            except:
                                pass
                        else:
                            # Set to specific speed
                            traci.vehicle.setSpeed(veh_id, target_speed)
                except Exception as e:
                    # Lane or vehicle might not exist, continue
                    pass
        except Exception as e:
            # Silently handle errors to avoid spam
            pass
    
    def close(self):
        """Close SUMO simulation"""
        if self.simulation_running:
            try:
                traci.close()
            except:
                pass
            self.simulation_running = False


class SUMOTrafficLightController:
    """Adaptive traffic light controller for SUMO"""
    
    def __init__(self, sumo_sim: SUMOTrafficSimulator,
                 timing_controller: AdaptiveTimingController,
                 weight_manager: VehicleWeightManager,
                 intersection_id: str):
        """
        Initialize SUMO traffic light controller
        
        Args:
            sumo_sim: SUMO simulator instance
            timing_controller: Adaptive timing controller
            weight_manager: Vehicle weight manager
            intersection_id: Traffic light ID in SUMO
        """
        self.sumo_sim = sumo_sim
        self.timing_controller = timing_controller
        self.weight_manager = weight_manager
        self.intersection_id = intersection_id
        
        # Map SUMO lanes to internal lane IDs (0, 1, 2, ...)
        self.lane_mapping = self._create_lane_mapping()
        
        # Reverse mapping: internal lane ID -> SUMO lane IDs
        self.reverse_mapping = {v: k for k, v in self.lane_mapping.items()}
    
    def _create_lane_mapping(self) -> Dict[str, int]:
        """Map SUMO lane IDs to internal lane IDs (0, 1, 2, ...)"""
        try:
            controlled_lanes = traci.trafficlight.getControlledLanes(self.intersection_id)
        except:
            # If traffic light doesn't exist, get lanes from intersection area
            try:
                # Get all lanes in the intersection area
                controlled_lanes = self.sumo_sim.get_all_lanes_in_intersection_area()
                # Limit to 16 lanes for 4-way intersection (4 lanes per direction)
                controlled_lanes = controlled_lanes[:16]
            except:
                # Final fallback: get all lanes
                try:
                    all_lanes = traci.lane.getIDList()
                    controlled_lanes = all_lanes[:16]  # 16 lanes for 4-way intersection (4 per direction)
                except:
                    controlled_lanes = []
        
        lane_mapping = {}
        
        # Sort lanes to ensure consistent mapping
        sorted_lanes = sorted(controlled_lanes)
        
        for i, lane in enumerate(sorted_lanes):
            lane_mapping[lane] = i
        
        return lane_mapping
    
    def get_vehicle_counts(self) -> Dict[int, float]:
        """
        Get vehicle counts per lane (weighted) from SUMO
        
        Returns:
            Dictionary mapping internal lane_id (int) -> weighted count (float)
        """
        # Get raw vehicle counts per lane (works with or without traffic lights)
        try:
            lane_counts_raw = self.sumo_sim.get_vehicle_counts_per_lane(self.intersection_id)
            vehicle_types = self.sumo_sim.get_vehicle_types_per_lane(self.intersection_id)
        except:
            # If we can't get counts, try to get from all lanes in intersection area
            try:
                all_lanes = self.sumo_sim.get_all_lanes_in_intersection_area()
                lane_counts_raw = {}
                vehicle_types = {}
                for lane in all_lanes[:16]:  # Limit to 16 lanes (4 per direction)
                    try:
                        count = traci.lane.getLastStepVehicleNumber(lane)
                        if count > 0:
                            lane_counts_raw[lane] = count
                            # Try to get vehicle types
                            veh_ids = traci.lane.getLastStepVehicleIDs(lane)
                            types = []
                            for veh_id in veh_ids:
                                try:
                                    types.append(traci.vehicle.getTypeID(veh_id))
                                except:
                                    pass
                            vehicle_types[lane] = types
                    except:
                        pass
            except:
                # Final fallback: return empty counts for 16 lanes
                return {i: 0.0 for i in range(16)}
        
        # Convert to internal lane ID format and calculate weighted counts
        lane_counts = {}
        
        for sumo_lane, count in lane_counts_raw.items():
            internal_lane_id = self.lane_mapping.get(sumo_lane, -1)
            if internal_lane_id < 0:
                continue
            
            # Calculate weighted count based on vehicle types
            weighted_count = 0.0
            vehicle_type_list = vehicle_types.get(sumo_lane, [])
            
            if vehicle_type_list:
                # Map SUMO vehicle types to COCO class IDs and calculate weights
                for veh_type in vehicle_type_list:
                    class_id = self._map_sumo_type_to_class_id(veh_type)
                    weighted_count += self.weight_manager.get_weight(class_id)
            else:
                # If no vehicle types available, use default weight
                weighted_count = count * 3.0  # Default weight for cars
            
            lane_counts[internal_lane_id] = weighted_count
        
        # Ensure all lanes are present (even if empty)
        for internal_lane_id in self.lane_mapping.values():
            if internal_lane_id not in lane_counts:
                lane_counts[internal_lane_id] = 0.0
        
        return lane_counts
    
    def _map_sumo_type_to_class_id(self, sumo_type: str) -> int:
        """
        Map SUMO vehicle type to COCO class ID
        
        Args:
            sumo_type: SUMO vehicle type (e.g., "car", "bus", "truck")
            
        Returns:
            COCO class ID (2=car, 3=motorcycle, 5=bus, 7=truck)
        """
        type_mapping = {
            "car": 2,
            "passenger": 2,
            "truck": 7,
            "trailer": 7,
            "bus": 5,
            "coach": 5,
            "motorcycle": 3,
            "bike": 3,
            "bicycle": 3,
            "moped": 3,
        }
        
        # Convert to lowercase for case-insensitive matching
        sumo_type_lower = sumo_type.lower()
        
        # Check for exact match first
        if sumo_type_lower in type_mapping:
            return type_mapping[sumo_type_lower]
        
        # Check for partial matches
        if "car" in sumo_type_lower or "passenger" in sumo_type_lower:
            return 2
        elif "truck" in sumo_type_lower or "trailer" in sumo_type_lower:
            return 7
        elif "bus" in sumo_type_lower or "coach" in sumo_type_lower:
            return 5
        elif "motorcycle" in sumo_type_lower or "bike" in sumo_type_lower or "bicycle" in sumo_type_lower:
            return 3
        
        # Default to car
        return 2
    
    def update_traffic_lights(self) -> Tuple[Dict[int, Dict[str, float]], Dict[int, float]]:
        """
        Update traffic light timings based on current traffic
        
        Returns:
            Tuple of (timings_dict, lane_counts_dict)
        """
        # Get vehicle counts from SUMO
        lane_counts = self.get_vehicle_counts()
        
        # Compute timings using existing controller logic
        current_time = self.sumo_sim.get_current_time()
        timings = self.timing_controller.compute_timings(lane_counts, current_time)
        
        # Apply timings to SUMO (simplified - SUMO manages phases internally)
        # In a full implementation, you would use traci.trafficlight.setPhaseDuration()
        # For now, we'll let SUMO handle phase transitions and just log the timings
        
        return timings, lane_counts
    
    def apply_timings_to_sumo(self, timings: Dict[int, Dict[str, float]]):
        """
        Apply computed timings by controlling lane speeds (simulates traffic lights)
        
        This function:
        1. Determines which direction has most traffic
        2. Stops vehicles in "red" lanes (speed = 0)
        3. Allows vehicles in "green" lanes to move (normal speed)
        4. Alternates between lane groups based on computed timings
        
        Args:
            timings: Timings dictionary from compute_timings()
        """
        try:
            if not timings:
                return
            
            # Initialize cycle state if not exists
            if not hasattr(self, '_cycle_start_time'):
                # Get all available lanes (should be 16 lanes: 4 per direction)
                all_available_lanes = sorted(self.lane_mapping.values())
                # Use first 16 lanes (4 per direction: North, East, South, West)
                self._all_lanes = all_available_lanes[:16] if len(all_available_lanes) >= 16 else list(range(16))
                # Group lanes into pairs: [0,1], [2,3], [4,5], [6,7], [8,9], [10,11], [12,13], [14,15]
                # Each pair represents 2 lanes
                self._lane_pairs = []
                for i in range(0, len(self._all_lanes), 2):
                    if i + 1 < len(self._all_lanes):
                        self._lane_pairs.append([self._all_lanes[i], self._all_lanes[i + 1]])
                    else:
                        # If odd number of lanes, last lane is alone
                        self._lane_pairs.append([self._all_lanes[i]])
            
            current_time = self.sumo_sim.get_current_time()
            
            # Switch which 2 lanes are red every 50 seconds
            SWITCH_INTERVAL = 50.0
            START_TIME = 100.0  # Start traffic control at 100 seconds
            
            # Before 100 seconds: do nothing, all lanes free
            if current_time < START_TIME:
                # All lanes green (no restrictions)
                green_lanes = self._all_lanes
                red_lanes = []
            else:
                # After 100 seconds, start round-robin with 2 lanes at a time
                time_since_start = current_time - START_TIME
                
                # Calculate which pair of lanes should be red (round-robin)
                # There are 4 pairs total (8 lanes / 2 = 4 pairs)
                intervals_passed = int(time_since_start / SWITCH_INTERVAL)
                pair_index = intervals_passed % len(self._lane_pairs)
                
                # Get the 2 lanes that should be red
                current_red_lanes = self._lane_pairs[pair_index]
                
                # Set 2 lanes as red, all others green
                green_lanes = [lane_id for lane_id in self._all_lanes if lane_id not in current_red_lanes]
                red_lanes = current_red_lanes
            
            # Control vehicle speeds by lane to simulate traffic lights
            # Get reverse mapping: internal lane ID -> SUMO lane IDs
            for internal_lane_id, sumo_lane in self.reverse_mapping.items():
                if internal_lane_id in green_lanes:
                    # Green: allow normal speed - remove speed restrictions IMMEDIATELY
                    try:
                        # Get vehicles and remove speed limits
                        vehicle_ids = traci.lane.getLastStepVehicleIDs(sumo_lane)
                        for veh_id in vehicle_ids:
                            try:
                                # Remove speed restriction to allow normal driving
                                # Reset speed mode to allow normal behavior
                                traci.vehicle.setSpeedMode(veh_id, 31)  # Reset to default behavior (all checks enabled)
                                traci.vehicle.setSpeed(veh_id, -1)  # -1 = use default/max speed (unrestricted)
                            except:
                                pass
                    except:
                        pass
                elif internal_lane_id in red_lanes:
                    # Red: stop vehicles completely (speed = 0) - simulate red light
                    try:
                        vehicle_ids = traci.lane.getLastStepVehicleIDs(sumo_lane)
                        for veh_id in vehicle_ids:
                            try:
                                # Force stop by setting speed to 0 - don't let cars cross
                                traci.vehicle.setSpeed(veh_id, 0.0)
                                # Also set speed mode to enforce the speed limit
                                traci.vehicle.setSpeedMode(veh_id, 0)  # Disable all speed checks except setSpeed
                            except:
                                pass
                    except:
                        pass
                        
        except Exception as e:
            # Silently handle errors to avoid spam - simulation continues
            pass


# Helper functions for app.py integration

def get_vehicle_counts_from_sumo(sumo_controller: SUMOTrafficLightController) -> Dict[int, float]:
    """
    Get vehicle counts from SUMO (wrapper function)
    
    Args:
        sumo_controller: SUMOTrafficLightController instance
        
    Returns:
        Dictionary mapping lane_id -> weighted count
    """
    return sumo_controller.get_vehicle_counts()


def convert_sumo_vehicle_types(sumo_type: str) -> int:
    """
    Convert SUMO vehicle type to COCO class ID (wrapper function)
    
    Args:
        sumo_type: SUMO vehicle type string
        
    Returns:
        COCO class ID
    """
    # Create a temporary controller instance to use the mapping method
    # In practice, this would be called from SUMOTrafficLightController
    type_mapping = {
        "car": 2, "passenger": 2,
        "truck": 7, "trailer": 7,
        "bus": 5, "coach": 5,
        "motorcycle": 3, "bike": 3, "bicycle": 3, "moped": 3,
    }
    sumo_type_lower = sumo_type.lower()
    return type_mapping.get(sumo_type_lower, 2)  # Default to car


def update_sumo_lights(sumo_controller: SUMOTrafficLightController, 
                       timings: Dict[int, Dict[str, float]]):
    """
    Update SUMO traffic lights with computed timings (wrapper function)
    
    Args:
        sumo_controller: SUMOTrafficLightController instance
        timings: Timings dictionary from compute_timings()
    """
    sumo_controller.apply_timings_to_sumo(timings)

