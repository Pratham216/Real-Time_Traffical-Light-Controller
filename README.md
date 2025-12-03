# 🚦 Smart Traffic Light Controller

An intelligent, adaptive traffic light control system that uses computer vision and machine learning to optimize traffic flow at intersections. The system dynamically adjusts traffic light timings based on real-time vehicle detection and tracking.

## 📋 Table of Contents

- [Features](#features)
- [Architecture](#architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Components](#components)
- [Traffic Light Control Algorithm](#traffic-light-control-algorithm)
- [Statistics and Analytics](#statistics-and-analytics)
- [Configuration](#configuration)
- [File Structure](#file-structure)
- [Requirements](#requirements)

## ✨ Features

### Core Functionality

1. **Real-Time Vehicle Detection**
   - YOLOv8-based object detection (supports nano, small, and medium models)
   - Detects cars, motorcycles, buses, and trucks
   - Custom fine-tuned model support for emergency vehicle detection
   - Configurable confidence thresholds

2. **Multi-Object Tracking**
   - ByteTrack algorithm for robust vehicle tracking
   - Maintains vehicle IDs across frames
   - Tracks vehicle positions and movements

3. **Dynamic Lane Detection**
   - Automatic lane detection using Hough Transform
   - Manual lane configuration support
   - Vehicle-to-lane assignment
   - Supports 1-6 lanes per direction

4. **Adaptive Traffic Light Control**
   - Dynamic timing adjustment based on vehicle density
   - Weighted vehicle counting (buses and trucks count more)
   - Emergency vehicle prioritization
   - Real-time phase optimization

5. **SUMO Simulation Integration**
   - Full SUMO traffic simulation support
   - Real-time traffic light control in SUMO
   - Vehicle count extraction from SUMO
   - Lane-based traffic management

6. **Comprehensive Statistics**
   - Latency tracking (processing time between frames)
   - Throughput measurement (vehicles processed per second)
   - Initial and final vehicle counts
   - Detailed CSV export with all metrics
   - Real-time performance graphs

## 🏗️ Architecture

The system consists of several modular components:

```
┌─────────────────┐
│   Video Input   │
│  (Camera/File)  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Vehicle        │
│  Detection      │
│  (YOLOv8)       │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Vehicle        │
│  Tracking       │
│  (ByteTrack)    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Lane           │
│  Detection      │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Vehicle Count  │
│  per Lane       │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Weighted       │
│  Counting       │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Adaptive       │
│  Timing         │
│  Controller     │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Traffic Light  │
│  Control        │
└─────────────────┘
```

## 📦 Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (optional, for faster inference)
- SUMO (for simulation mode)

### Step 1: Clone the Repository

```bash
git clone <repository-url>
cd traffic-test
```

### Step 2: Install Python Dependencies

```bash
pip install -r requirements.txt
```

### Step 3: Install SUMO (Optional, for Simulation Mode)

**Windows:**
- Download from [SUMO Download Page](https://sumo.dlr.de/docs/Downloads.php)
- Add SUMO to your system PATH

**Linux:**
```bash
sudo apt-get install sumo sumo-tools sumo-doc
```

**macOS:**
```bash
brew install sumo
```

### Step 4: Download YOLOv8 Models (Optional)

The system will automatically download models on first use, or you can download them manually:
- `yolov8n.pt` - Nano (fastest, least accurate)
- `yolov8s.pt` - Small (balanced, recommended)
- `yolov8m.pt` - Medium (most accurate, slower)

## 🚀 Usage

### Starting the Application

```bash
streamlit run app.py
```

The application will open in your default web browser at `http://localhost:8501`

### Operation Modes

#### 1. Real Camera Mode

Process video files or camera feeds:

1. Select **"Real Camera"** mode from the sidebar
2. Upload a video file (MP4, AVI, MOV)
3. Configure detection and timing parameters
4. Click **"Start Processing"**
5. View real-time results and statistics
6. Save statistics to CSV when done

#### 2. SUMO Simulation Mode

Run traffic simulations with adaptive control:

1. Select **"SUMO Simulation"** mode from the sidebar
2. Specify SUMO configuration file path
3. Configure simulation parameters
4. Click **"Start SUMO Simulation"**
5. Monitor real-time traffic flow and light timings
6. Stop simulation anytime with **"Stop Simulation"** button
7. View comprehensive results and graphs
8. Export statistics to CSV

## 🔧 Components

### 1. Vehicle Detection (`detection.py`)

- **YOLOv8 Integration**: Uses Ultralytics YOLOv8 for object detection
- **Vehicle Classes**: Detects cars (class 2), motorcycles (class 3), buses (class 5), trucks (class 7)
- **Custom Models**: Supports fine-tuned models for emergency vehicle detection
- **Device Support**: CUDA GPU acceleration or CPU fallback

### 2. Vehicle Tracking (`tracking.py`)

- **ByteTrack Algorithm**: Multi-object tracking via Ultralytics
- **Track Management**: Maintains consistent vehicle IDs across frames
- **Position Tracking**: Tracks vehicle bottom-center positions for lane assignment
- **Emergency Detection**: Identifies emergency vehicles from custom models

### 3. Lane Detection (`lanes.py`)

- **Hough Transform**: Detects lane markings using edge detection
- **Auto-Detection**: Automatically determines number of lanes
- **Manual Configuration**: Supports manual lane count specification
- **ROI-Based**: Uses Region of Interest for improved accuracy
- **Vehicle Assignment**: Assigns vehicles to lanes based on position

### 4. Vehicle Weighting (`vehicle_weights.py`)

- **Weighted Counting**: Different vehicle types have different weights
- **Default Weights**:
  - Car: 3.0
  - Motorcycle: 2.0
  - Bus: 5.0
  - Truck: 4.0
  - Ambulance: 10.0 (highest priority)
  - Fire Truck: 10.0 (highest priority)
  - Police: 8.0 (very high priority)
- **Customizable**: All weights can be adjusted via UI

### 5. Adaptive Timing Controller (`timing.py`)

- **Adaptive Formula**: `Gi = min(max(Gmin, G0 + k * Ni), Gmax)`
  - `Gi`: Green time for lane i
  - `G0`: Base green time
  - `k`: Scaling factor (seconds per vehicle)
  - `Ni`: Vehicle count (weighted or raw) for lane i
  - `Gmin`: Minimum green time
  - `Gmax`: Maximum green time
- **Dynamic Cycle Time**: Adjusts based on traffic density
- **Phase Management**: Manages green, yellow, and red phases

### 6. SUMO Integration (`sumo_integration.py`)

- **TraCI Communication**: Real-time communication with SUMO via TraCI
- **Lane Mapping**: Maps SUMO lanes to internal lane IDs
- **Vehicle Counting**: Extracts vehicle counts per lane from SUMO
- **Traffic Light Control**: Applies adaptive timings to SUMO traffic lights
- **Speed Control**: Uses lane-based speed control when traffic lights unavailable

### 7. Statistics Logger (`utils.py`)

- **Frame-by-Frame Logging**: Records all metrics for each frame/step
- **Performance Metrics**: Tracks latency and throughput
- **Vehicle Counts**: Records initial and final vehicle counts
- **CSV Export**: Exports detailed data and summary statistics
- **Graph Generation**: Creates visualizations for analysis

## 🚦 Traffic Light Control Algorithm

### Current Implementation

The system uses an **adaptive timing algorithm** that dynamically adjusts green light durations based on vehicle density:

#### Algorithm Overview

1. **Vehicle Detection & Counting**
   - Detects and tracks vehicles in each lane
   - Applies vehicle weights (buses/trucks count more)
   - Prioritizes emergency vehicles

2. **Green Time Calculation**
   - For each lane i: `Gi = min(max(Gmin, G0 + k * Ni), Gmax)`
   - Where:
     - `Ni` = weighted vehicle count in lane i
     - `G0` = base green time (default: 5 seconds)
     - `k` = scaling factor (default: 1.0 seconds/vehicle)
     - `Gmin` = minimum green time (default: 5 seconds)
     - `Gmax` = maximum green time (default: 60 seconds)

3. **Cycle Time Calculation**
   - Total cycle time = max(Gi) + yellow_time + all_red_time
   - Ensures all lanes complete their green phase

4. **Phase Sequencing**
   - Each lane gets its calculated green time
   - Yellow phase: fixed duration (default: 3 seconds)
   - All-red phase: fixed duration (default: 1 second)
   - Red time: calculated as cycle_time - green_time - yellow_time - all_red_time

### Recommended Implementation

For a production system, the following enhancements should be implemented:

#### 1. Priority-Based Phase Selection

Instead of fixed sequencing, select phases based on priority:

```python
def select_next_phase(lane_counts, current_phase, timings):
    """
    Select next phase based on vehicle density and priority
    
    Priority factors:
    - Vehicle count (weighted)
    - Waiting time
    - Emergency vehicles present
    - Queue length
    """
    priorities = {}
    for lane_id, count in lane_counts.items():
        waiting_time = get_waiting_time(lane_id)
        emergency_present = check_emergency_vehicles(lane_id)
        
        priority = (
            count * 0.4 +                    # Vehicle density (40%)
            waiting_time * 0.3 +              # Waiting time (30%)
            emergency_present * 100 * 0.3    # Emergency priority (30%)
        )
        priorities[lane_id] = priority
    
    # Select lane with highest priority
    next_lane = max(priorities, key=priorities.get)
    return create_phase_for_lane(next_lane, timings)
```

#### 2. Conflict-Free Phase Management

Ensure no conflicting lanes get green simultaneously:

```python
def create_conflict_free_phases(lanes, conflicts):
    """
    Create phases ensuring no conflicts
    
    Args:
        lanes: List of all lanes
        conflicts: Dictionary mapping lane -> list of conflicting lanes
    
    Returns:
        List of phase groups (lanes that can be green together)
    """
    phase_groups = []
    remaining_lanes = set(lanes)
    
    while remaining_lanes:
        # Find maximum independent set
        current_phase = []
        for lane in sorted(remaining_lanes, key=lambda x: lane_counts[x], reverse=True):
            # Check if lane conflicts with any lane in current phase
            if not any(lane in conflicts.get(conflict_lane, []) 
                      for conflict_lane in current_phase):
                current_phase.append(lane)
        
        phase_groups.append(current_phase)
        remaining_lanes -= set(current_phase)
    
    return phase_groups
```

#### 3. Dynamic Cycle Time Optimization

Adjust cycle time based on overall traffic:

```python
def optimize_cycle_time(lane_counts, base_cycle, min_cycle, max_cycle):
    """
    Optimize cycle time based on total traffic
    
    Formula: C = C_base * (1 + α * (N_total / N_threshold))
    """
    total_vehicles = sum(lane_counts.values())
    threshold = 20  # Threshold for normal traffic
    alpha = 0.1     # Scaling factor
    
    if total_vehicles < threshold:
        # Light traffic: shorter cycles
        cycle_time = base_cycle * (1 - alpha)
    else:
        # Heavy traffic: longer cycles
        cycle_time = base_cycle * (1 + alpha * (total_vehicles / threshold))
    
    return max(min_cycle, min(cycle_time, max_cycle))
```

#### 4. Emergency Vehicle Preemption

Immediate priority for emergency vehicles:

```python
def check_emergency_preemption(lanes, emergency_vehicles):
    """
    Check if emergency vehicle requires immediate green
    
    Returns:
        Lane ID that should get immediate green, or None
    """
    for lane_id, vehicles in emergency_vehicles.items():
        if vehicles:
            # Emergency vehicle detected: immediate green
            return lane_id
    return None

def apply_emergency_preemption(current_phase, emergency_lane):
    """
    Interrupt current phase for emergency vehicle
    """
    if emergency_lane:
        # Immediately switch to emergency lane
        # Shorten current green phase if needed
        # Skip to emergency lane green phase
        return create_emergency_phase(emergency_lane)
    return current_phase
```

#### 5. Queue Length Estimation

Consider queue length, not just visible vehicles:

```python
def estimate_queue_length(lane_id, current_count, historical_data):
    """
    Estimate total queue length including vehicles not yet visible
    
    Uses historical data and vehicle arrival rates
    """
    arrival_rate = calculate_arrival_rate(lane_id, historical_data)
    waiting_time = get_average_waiting_time(lane_id)
    
    # Estimate: visible + (arrival_rate * waiting_time)
    estimated_queue = current_count + (arrival_rate * waiting_time)
    return estimated_queue
```

#### 6. Predictive Timing

Use historical patterns for better timing:

```python
def predictive_timing(lane_id, current_count, time_of_day, day_of_week):
    """
    Use historical patterns to predict optimal timing
    
    Considers:
    - Time of day patterns
    - Day of week patterns
    - Current traffic conditions
    """
    historical_pattern = get_historical_pattern(lane_id, time_of_day, day_of_week)
    predicted_demand = historical_pattern * current_count
    
    # Adjust timing based on predicted demand
    return calculate_timing(predicted_demand)
```

## 📊 Statistics and Analytics

### Metrics Tracked

1. **Latency (ms)**
   - Processing time between frames/steps
   - Average, minimum, and maximum latency
   - Helps identify performance bottlenecks

2. **Throughput (vehicles/second)**
   - Vehicles processed per second
   - System efficiency metric
   - Average and maximum throughput

3. **Vehicle Counts**
   - Initial vehicle count (start of simulation)
   - Final vehicle count (end of simulation)
   - Per-lane vehicle counts over time
   - Total vehicles processed

4. **Timing Data**
   - Green, yellow, and red times per lane
   - Cycle times
   - Phase transitions

5. **Performance Graphs**
   - Latency over time
   - Throughput over time
   - Total vehicles over time
   - Lane-wise vehicle counts

### CSV Export

Two CSV files are generated:

1. **Detailed Data CSV** (`traffic_stats_YYYYMMDD_HHMMSS.csv`)
   - Frame-by-frame data
   - All metrics for each time step
   - Lane-specific counts and timings

2. **Summary Statistics CSV** (`traffic_stats_YYYYMMDD_HHMMSS_summary.csv`)
   - Overall statistics
   - Performance metrics
   - Initial/final counts
   - Average, min, max values

## ⚙️ Configuration

### Detection Parameters

- **Model Size**: nano (fastest), small (balanced), medium (most accurate)
- **Confidence Threshold**: 0.1 - 0.9 (lower = more detections, may include false positives)
- **Device**: CUDA (GPU) or CPU

### Lane Detection

- **Number of Lanes**: 1-6 lanes per direction
- **ROI Ratio**: 0.3 - 0.9 (portion of frame height for detection)

### Vehicle Weights

Customizable weights for each vehicle type:
- Car: 1.0 - 10.0 (default: 3.0)
- Motorcycle: 1.0 - 10.0 (default: 2.0)
- Bus: 1.0 - 10.0 (default: 5.0)
- Truck: 1.0 - 10.0 (default: 4.0)
- Emergency vehicles: 5.0 - 20.0 (default: 8.0 - 10.0)

### Timing Parameters

- **Base Green Time**: 3.0 - 20.0 seconds (default: 5.0)
- **Scaling Factor**: 0.1 - 5.0 seconds/vehicle (default: 1.0)
- **Min Green Time**: 3.0 - 15.0 seconds (default: 5.0)
- **Max Green Time**: 20.0 - 120.0 seconds (default: 60.0)
- **Yellow Time**: 1.0 - 5.0 seconds (default: 3.0)
- **All-Red Time**: 0.5 - 3.0 seconds (default: 1.0)

### Performance Settings

- **Target FPS**: 5 - 60 (default: 30)
- **Frame Skip**: Process every N frames (1-5, default: 1)
- **Resize Video**: Resize to 640x480 for faster processing

## 📁 File Structure

```
traffic-test/
├── app.py                      # Main Streamlit application
├── detection.py                # YOLOv8 vehicle detection
├── tracking.py                 # ByteTrack vehicle tracking
├── lanes.py                    # Lane detection module
├── timing.py                   # Adaptive timing controller
├── vehicle_weights.py         # Vehicle weight management
├── utils.py                    # Utilities and statistics logger
├── sumo_integration.py         # SUMO simulation integration
├── requirements.txt            # Python dependencies
├── README.md                   # This file
│
├── sumo_scenarios/             # SUMO scenario files
│   └── intersection/
│       ├── intersection.sumocfg
│       ├── intersection.net.xml
│       ├── intersection.rou.xml
│       └── intersection.add.xml
│
└── [Model files]               # YOLOv8 model weights (.pt files)
```

## 📋 Requirements

### Python Packages

```
streamlit>=1.28.0
opencv-python>=4.8.0
ultralytics>=8.0.0
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.3.0
scipy>=1.11.0
```

### Optional Dependencies

```
traci>=1.18.0          # SUMO TraCI (for simulation mode)
sumolib>=1.18.0        # SUMO library (for simulation mode)
```

### System Requirements

- **CPU**: Multi-core processor recommended
- **RAM**: 4GB minimum, 8GB recommended
- **GPU**: NVIDIA GPU with CUDA support (optional, for faster inference)
- **Storage**: 500MB for application + model files (~500MB per YOLOv8 model)

## 🎯 Use Cases

1. **Traffic Flow Optimization**: Reduce waiting times at intersections
2. **Emergency Vehicle Priority**: Automatically prioritize emergency vehicles
3. **Traffic Analysis**: Analyze traffic patterns and optimize infrastructure
4. **Simulation Testing**: Test traffic light strategies before real-world deployment
5. **Research & Development**: Study adaptive traffic control algorithms

## 🔮 Future Enhancements

- [ ] Multi-intersection coordination
- [ ] Pedestrian detection and timing
- [ ] Weather-aware timing adjustments
- [ ] Machine learning-based phase prediction
- [ ] Real-time traffic flow prediction
- [ ] Integration with traffic management systems
- [ ] Mobile app for monitoring
- [ ] Cloud deployment support

## 📝 License

[Specify your license here]

## 👥 Contributors

[Add contributor names]

## 📧 Contact

[Add contact information]

---

**Note**: This system is designed for research and educational purposes. For production deployment, ensure compliance with local traffic regulations and conduct thorough testing.
