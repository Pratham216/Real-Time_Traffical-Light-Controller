"""
Vehicle Weight Configuration Module
Defines weights for different vehicle types for adaptive timing calculation
"""
from typing import Dict, Optional

# COCO class indices mapping
# car=2, motorcycle=3, bus=5, truck=7
COCO_CLASS_NAMES = {
    2: "car",
    3: "motorcycle", 
    5: "bus",
    7: "truck"
}

# Default vehicle weights
# Higher weight = more priority in timing calculation
DEFAULT_VEHICLE_WEIGHTS = {
    "car": 3.0,           # Normal weight for cars
    "motorcycle": 2.0,    # Lower weight for motorcycles/bikes
    "bus": 5.0,           # Higher weight (carries more people)
    "truck": 4.0,         # Medium-high weight (could be emergency vehicle)
    # Emergency vehicles (highest priority)
    "ambulance": 10.0,    # Highest priority - medical emergency
    "fire_truck": 10.0,   # Highest priority - fire emergency
    "police": 8.0         # Very high priority - law enforcement
}

# Vehicle type categories
VEHICLE_CATEGORIES = {
    "emergency": ["ambulance", "police", "fire_truck"],  # For future use
    "public_transport": ["bus"],
    "personal": ["car", "motorcycle"],
    "commercial": ["truck"]
}


class VehicleWeightManager:
    """Manages vehicle weights for timing calculation"""
    
    def __init__(self, custom_weights: Optional[Dict[str, float]] = None):
        """
        Initialize weight manager
        
        Args:
            custom_weights: Optional dictionary to override default weights
                           e.g., {"car": 3.5, "bus": 6.0}
        """
        self.weights = DEFAULT_VEHICLE_WEIGHTS.copy()
        
        # Override with custom weights if provided
        if custom_weights:
            self.weights.update(custom_weights)
    
    def get_weight(self, class_id: int, emergency_type: Optional[str] = None) -> float:
        """
        Get weight for a vehicle class ID, with optional emergency vehicle override
        
        Args:
            class_id: COCO class ID (2=car, 3=motorcycle, 5=bus, 7=truck)
            emergency_type: Optional emergency vehicle type ('ambulance', 'fire_truck', 'police')
            
        Returns:
            Weight value for the vehicle type
        """
        # If emergency vehicle detected, use emergency weight
        if emergency_type and emergency_type in self.weights:
            return self.weights[emergency_type]
        
        # Otherwise use standard vehicle weight
        vehicle_type = COCO_CLASS_NAMES.get(class_id, "car")
        return self.weights.get(vehicle_type, 3.0)  # Default to 3.0 if unknown
    
    def get_vehicle_type(self, class_id: int) -> str:
        """
        Get vehicle type name from class ID
        
        Args:
            class_id: COCO class ID
            
        Returns:
            Vehicle type name
        """
        return COCO_CLASS_NAMES.get(class_id, "car")
    
    def set_weight(self, vehicle_type: str, weight: float):
        """
        Set weight for a specific vehicle type
        
        Args:
            vehicle_type: Vehicle type name (e.g., "car", "bus")
            weight: New weight value
        """
        if vehicle_type in self.weights:
            self.weights[vehicle_type] = weight
    
    def get_all_weights(self) -> Dict[str, float]:
        """Get all current weights"""
        return self.weights.copy()
    
    def calculate_weighted_count(self, vehicles: list) -> float:
        """
        Calculate weighted count from a list of vehicles
        
        Args:
            vehicles: List of vehicle dictionaries with 'class_id' key
            
        Returns:
            Total weighted count
        """
        total_weight = 0.0
        for vehicle in vehicles:
            if isinstance(vehicle, dict):
                class_id = vehicle.get('class_id')
                if class_id is not None:
                    total_weight += self.get_weight(class_id)
            elif isinstance(vehicle, (int, float)):
                # If vehicle is just a count, use default weight
                total_weight += vehicle * 3.0
        
        return total_weight

