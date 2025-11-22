# 🚦 Smart Traffic Light Controller

An intelligent traffic light control system that uses YOLOv8 object detection, multi-object tracking, and SUMO simulation to adaptively control traffic signal timings based on real-time vehicle counts.

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Latest-red.svg)](https://streamlit.io/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-green.svg)](https://github.com/ultralytics/ultralytics)
[![SUMO](https://img.shields.io/badge/SUMO-Integration-orange.svg)](https://www.eclipse.org/sumo/)

---

## 📋 Table of Contents

- [Features](#-features)
- [Architecture](#-architecture)
- [Installation](#-installation)
- [Usage](#-usage)
- [Technical Details](#-technical-details)
- [Project Structure](#-project-structure)
- [Configuration](#-configuration)
- [Performance](#-performance)
- [Future Enhancements](#-future-enhancements)
- [Contributing](#-contributing)
- [License](#-license)

---

## ✨ Features

### 🎥 Video-Based Traffic Analysis
- **YOLOv8 Vehicle Detection**: Real-time detection of cars, motorcycles, buses, and trucks
  - Model options: YOLOv8n (nano), YOLOv8s (small - default), YOLOv8m (medium)
  - GPU acceleration with FP16 precision support
  - Custom fine-tuned model support for emergency vehicles
  - Optimized confidence threshold (0.25) for better motorcycle detection

- **Multi-Object Tracking**: ByteTrack integration
  - Maintains consistent vehicle IDs across frames
  - Prevents double-counting
  - Tracks vehicle trajectories
  - Supports emergency vehicle detection (ambulance, fire truck, police)

- **Dynamic Lane Detection**: Automatic lane identification
  - Hough Transform-based lane boundary detection
  - Canny edge detection and line clustering
  - Vehicle-to-lane assignment
  - Supports 1-6 lanes (configurable)

- **Adaptive Timing Algorithm**: Dynamic traffic light timing
  - Formula: `Green Time = min(max(Min Green, Base Green + (Scaling Factor × Vehicle Count)), Max Green)`
  - Configurable parameters: base green, scaling factor, min/max green times
  - Fixed yellow and all-red intervals for safety
  - Per-lane timing calculation

- **Vehicle Weighting System**: Priority-based traffic control
  - Different weights for different vehicle types (cars, buses, trucks, motorcycles)
  - Emergency vehicle prioritization
  - Weighted vehicle counting for more accurate traffic assessment

### 🚗 SUMO Traffic Simulation Integration
- **Full SUMO Integration**: Complete traffic simulation environment
  - Real-time traffic simulation with SUMO
  - Integrated adaptive traffic light controller
  - Lane-based traffic control system
  - Supports multiple lanes for comprehensive intersection control
  - Dynamic traffic signal management based on real-time vehicle detection

- **Traffic Light Controller Integration**:
  - Seamless integration of adaptive timing algorithm with SUMO simulation
  - Real-time vehicle detection and counting from SUMO simulation
  - Automatic traffic light phase control based on traffic density
  - Configurable traffic control cycles
  - Support for complex intersection scenarios

- **Vehicle Speed Control**:
  - Red lanes: Vehicles stopped (speed = 0)
  - Green lanes: Vehicles move freely (speed unrestricted)
  - Smooth transitions between traffic light phases

### 🖥️ User Interface (Streamlit)
- **Dual Mode Operation**:
  - **Real Camera Mode**: Process uploaded traffic videos
  - **SUMO Simulation Mode**: Run traffic simulations with adaptive control

- **Configuration Options**:
  - Model selection (Pretrained COCO or Custom Fine-Tuned)
  - Model size selection (nano/small/medium)
  - Confidence threshold adjustment
  - Device selection (CPU/CUDA)
  - Lane detection settings
  - Vehicle weight parameters
  - Emergency vehicle detection toggle
  - Timing parameters (base green, scaling factor, min/max green)
  - Performance settings (FPS, frame skip, resize)

- **Control Features**:
  - Start/Stop simulation buttons
  - Reset functionality
  - Save statistics to CSV
  - Real-time video display with annotations
  - Live timing table updates
  - Statistics dashboard

---

## 🏗️ Architecture

### Core Modules

| Module | Description |
|--------|-------------|
| `app.py` | Main Streamlit application - User interface and controls |
| `detection.py` | YOLOv8 vehicle detection - Model loading and inference |
| `tracking.py` | Multi-object tracking - ByteTrack integration |
| `lanes.py` | Lane detection - Hough Transform lane detection |
| `timing.py` | Adaptive timing controller - Timing algorithm implementation |
| `sumo_integration.py` | SUMO simulation - Traffic light controller integration |
| `utils.py` | Utility functions - Traffic logging and data formatting |
| `vehicle_weights.py` | Vehicle weighting system - Weight management |

---

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- CUDA-capable GPU (optional, for faster processing)

### Step 1: Clone the Repository
```bash
git clone https://github.com/Pratham216/Real-Time_Traffical-Light-Controller.git
cd Real-Time_Traffical-Light-Controller
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Install SUMO (for simulation mode)
- Download SUMO from [https://www.eclipse.org/sumo/](https://www.eclipse.org/sumo/)
- Add SUMO to your system PATH
- Verify installation: `sumo --version`

### Step 4: Download YOLOv8 Models
The models (`.pt` files) are excluded from the repository. You can:
- Download them automatically (they'll be downloaded on first run)
- Or download manually from [Ultralytics](https://github.com/ultralytics/ultralytics)

---

## 💻 Usage

### Running the Application

```bash
streamlit run app.py
```

The application will open in your default web browser at `http://localhost:8501`

### Basic Workflow

1. **Select Mode**:
   - Choose "Real Camera Mode" for video processing
   - Choose "SUMO Simulation Mode" for traffic simulation

2. **Configure Settings**:
   - Select model size (nano/small/medium)
   - Adjust confidence threshold
   - Set lane detection parameters
   - Configure timing parameters

3. **Start Processing**:
   - Upload a video file (for Real Camera Mode)
   - Click "Start Simulation" (for SUMO Mode)

4. **View Results**:
   - Real-time video display with annotations
   - Live statistics dashboard
   - Download statistics as CSV

---

## 🔧 Technical Details

### Model Support
- **YOLOv8n.pt**: Nano model (fastest, least accurate)
- **YOLOv8s.pt**: Small model (balanced, default) ⭐
- **YOLOv8m.pt**: Medium model (most accurate, slower)
- **Custom Models**: Support for fine-tuned models with emergency vehicles

### Vehicle Classes Detected
| Class | ID | Description |
|-------|----|----|
| Car | 2 | Standard passenger vehicles |
| Motorcycle | 3 | Motorcycles and scooters |
| Bus | 5 | Buses and large passenger vehicles |
| Truck | 7 | Trucks and commercial vehicles |
| Emergency | 8-10 | Ambulance, Fire Truck, Police (custom) |

### Traffic Control Parameters

| Parameter | Default Value | Description |
|-----------|---------------|-------------|
| Base Green Time | 5.0 seconds | Minimum green time per cycle |
| Scaling Factor | 1.0 sec/vehicle | Additional time per vehicle |
| Min Green Time | 5.0 seconds | Absolute minimum green time |
| Max Green Time | 60.0 seconds | Maximum allowed green time |
| Yellow Time | 3.0 seconds | Fixed yellow interval |
| All-Red Time | 1.0 seconds | Fixed all-red clearance time |

### SUMO Integration Features

| Feature | Description |
|---------|-------------|
| Real-time Detection | Vehicle detection and counting from SUMO simulation |
| Adaptive Control | Traffic light timing adjusts based on vehicle density |
| Multi-lane Support | Handles complex intersections with multiple lanes |
| Phase Management | Automatic traffic light phase transitions |
| Cycle Control | Configurable traffic light cycle management |

---

## 📁 Project Structure

```
Real-Time_Traffical-Light-Controller/
├── app.py                      # Main Streamlit application
├── detection.py                # YOLOv8 vehicle detection
├── tracking.py                 # ByteTrack multi-object tracking
├── lanes.py                    # Dynamic lane detection
├── timing.py                   # Adaptive timing calculation
├── sumo_integration.py         # SUMO simulation integration
├── utils.py                    # Helper functions and logging
├── vehicle_weights.py          # Vehicle weighting system
├── emergency_detector.py       # Emergency vehicle detection
├── requirements.txt            # Python dependencies
├── README.md                   # Project documentation
├── .gitignore                  # Git ignore rules
└── sumo_scenarios/             # SUMO simulation scenarios
    └── intersection/
        ├── intersection.sumocfg
        ├── intersection.net.xml
        ├── intersection.rou.xml
        └── intersection.add.xml
```

> **Note**: Model files (`.pt`) are excluded from the repository via `.gitignore`

---

## ⚙️ Configuration

### Environment Variables
You can configure the system using environment variables or through the Streamlit UI:

- `CUDA_VISIBLE_DEVICES`: Specify GPU device (e.g., "0" for first GPU)
- `SUMO_HOME`: Path to SUMO installation (if not in PATH)

### Model Configuration
Edit the model settings in `app.py` or use the UI:
- Default model: `yolov8s.pt`
- Confidence threshold: `0.25`
- Device: Auto-detect (CPU/CUDA)

---

## 📊 Performance

### Detection Performance
- **Model**: YOLOv8s (small) - balanced accuracy and speed
- **Confidence Threshold**: 0.25 (optimized for small vehicles)
- **Device Support**: CPU and CUDA (GPU)
- **Frame Rate**: Configurable (5-30 FPS)

### Simulation Performance
- **Lanes Supported**: Multiple lanes per intersection
- **Real-time Control**: Dynamic traffic light control based on vehicle detection
- **Adaptive Timing**: Traffic signals adjust automatically to traffic conditions
- **Integration**: Seamless connection between SUMO simulation and traffic controller

### System Requirements
- **Minimum**: CPU-only processing (slower)
- **Recommended**: CUDA-capable GPU for real-time processing
- **RAM**: 4GB minimum, 8GB recommended
- **Storage**: 2GB for models and dependencies

---

## 🔮 Future Enhancements

- [ ] Real-time webcam input support
- [ ] Multiple intersection control
- [ ] Traffic flow prediction using historical data
- [ ] Advanced phase scheduling algorithms
- [ ] Machine learning-based traffic pattern recognition
- [ ] Integration with real traffic light hardware
- [ ] Mobile app interface
- [ ] Cloud deployment support
- [ ] REST API for external integrations
- [ ] Real-time dashboard with WebSocket support

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

### How to Contribute
1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📝 License

This project is open source and available under the [MIT License](LICENSE).

---

## 🙏 Acknowledgments

- [Ultralytics](https://github.com/ultralytics/ultralytics) for YOLOv8
- [ByteTrack](https://github.com/ifzhang/ByteTrack) for multi-object tracking
- [SUMO](https://www.eclipse.org/sumo/) for traffic simulation
- [Streamlit](https://streamlit.io/) for the web interface

---

## 📧 Contact

For questions, suggestions, or support, please open an issue on GitHub.

---

**Made with ❤️ for smarter traffic management**
