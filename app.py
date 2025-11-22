"""
Smart Traffic Light Controller - Streamlit Application
Main entry point for the traffic light control system
"""
import streamlit as st
import cv2
import numpy as np
import tempfile
import os
from pathlib import Path
import time
import pandas as pd
from datetime import datetime

from detection import VehicleDetector
from tracking import VehicleTracker
from lanes import LaneDetector
from timing import AdaptiveTimingController
from vehicle_weights import VehicleWeightManager
from utils import TrafficLogger, draw_vehicle_boxes, count_vehicles_per_lane, \
                  get_lane_counts, get_weighted_lane_counts, format_timing_table, resize_frame

# SUMO integration (optional import)
try:
    from sumo_integration import SUMOTrafficSimulator, SUMOTrafficLightController, \
                                get_vehicle_counts_from_sumo, update_sumo_lights
    SUMO_AVAILABLE = True
except ImportError:
    SUMO_AVAILABLE = False
    SUMOTrafficSimulator = None
    SUMOTrafficLightController = None

# Page configuration
st.set_page_config(
    page_title="Smart Traffic Light Controller",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling (only affects Real Camera mode UI)
st.markdown("""
    <style>
    /* Main container styling */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    
    /* Header styling */
    h1 {
        color: #1f77b4;
        border-bottom: 3px solid #1f77b4;
        padding-bottom: 0.5rem;
    }
    
    h2, h3 {
        color: #2c3e50;
    }
    
    /* Metric cards */
    [data-testid="stMetricValue"] {
        font-size: 2rem;
        font-weight: bold;
    }
    
    /* Button styling */
    .stButton > button {
        width: 100%;
        border-radius: 10px;
        border: none;
        padding: 0.5rem 1rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    
    /* Dataframe styling */
    .dataframe {
        border-radius: 10px;
        overflow: hidden;
    }
    
    /* Info boxes */
    .stInfo {
        border-left: 4px solid #1f77b4;
        border-radius: 5px;
    }
    
    /* Success messages */
    .stSuccess {
        border-left: 4px solid #00cc00;
        border-radius: 5px;
    }
    
    /* Warning messages */
    .stWarning {
        border-left: 4px solid #ff9900;
        border-radius: 5px;
    }
    
    /* Error messages */
    .stError {
        border-left: 4px solid #ff0000;
        border-radius: 5px;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'video_processed' not in st.session_state:
    st.session_state.video_processed = False
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False
if 'video_file' not in st.session_state:
    st.session_state.video_file = None
if 'simulation_stopped' not in st.session_state:
    st.session_state.simulation_stopped = False
# For Real Camera mode UI improvements
if 'traffic_history' not in st.session_state:
    st.session_state.traffic_history = {'time': [], 'lane_counts': {}, 'total_vehicles': []}
if 'processing_active' not in st.session_state:
    st.session_state.processing_active = False
if 'stop_processing' not in st.session_state:
    st.session_state.stop_processing = False


# Helper functions for Real Camera mode UI improvements
def update_traffic_history(lane_counts, current_time, max_points=100):
    """Update traffic history for real-time graphs"""
    if 'traffic_history' not in st.session_state:
        st.session_state.traffic_history = {'time': [], 'lane_counts': {}, 'total_vehicles': []}
    
    history = st.session_state.traffic_history
    history['time'].append(current_time)
    history['total_vehicles'].append(sum(lane_counts.values()))
    
    # Get all known lane IDs (both existing and new)
    all_lane_ids = set(history['lane_counts'].keys()) | set(lane_counts.keys())
    
    # Update lane counts - ensure all lanes have the same length
    for lane_id in all_lane_ids:
        if lane_id not in history['lane_counts']:
            # Initialize new lane with zeros to match current time array length
            history['lane_counts'][lane_id] = [0] * len(history['time'])
        
        if lane_id in lane_counts:
            # Add current count
            history['lane_counts'][lane_id].append(lane_counts[lane_id])
        else:
            # Lane not in current counts, append 0 to keep arrays in sync
            history['lane_counts'][lane_id].append(0)
    
    # Keep only last max_points - ensure all arrays stay in sync
    if len(history['time']) > max_points:
        history['time'] = history['time'][-max_points:]
        history['total_vehicles'] = history['total_vehicles'][-max_points:]
        for lane_id in history['lane_counts']:
            if len(history['lane_counts'][lane_id]) > max_points:
                history['lane_counts'][lane_id] = history['lane_counts'][lane_id][-max_points:]


def create_traffic_graph():
    """Create real-time traffic count graph"""
    if 'traffic_history' not in st.session_state or not st.session_state.traffic_history['time']:
        st.info("Waiting for traffic data...")
        return
    
    history = st.session_state.traffic_history
    df = pd.DataFrame({
        'Time (s)': history['time'],
        'Total Vehicles': history['total_vehicles']
    })
    
    st.line_chart(df.set_index('Time (s)'), use_container_width=True, height=200)


def create_lane_traffic_graph():
    """Create lane-wise traffic count graph"""
    if 'traffic_history' not in st.session_state or not st.session_state.traffic_history['time']:
        return
    
    history = st.session_state.traffic_history
    if not history['lane_counts']:
        return
    
    # Get the time array length as reference
    time_length = len(history['time'])
    
    # Create DataFrame with all lane data, ensuring all arrays are the same length
    data = {'Time (s)': history['time']}
    for lane_id, counts in history['lane_counts'].items():
        # Ensure counts array matches time array length
        if len(counts) < time_length:
            # Pad with zeros if shorter
            counts = counts + [0] * (time_length - len(counts))
        elif len(counts) > time_length:
            # Truncate if longer
            counts = counts[:time_length]
        data[f'Lane {lane_id}'] = counts
    
    df = pd.DataFrame(data)
    st.line_chart(df.set_index('Time (s)'), use_container_width=True, height=200)


def display_timing_table_with_icons(timings):
    """Display timing table with icons for Red, Green, Yellow"""
    timing_df = format_timing_table(timings)
    
    # Add icons to the timing columns
    timing_df_display = timing_df.copy()
    timing_df_display['Green (s)'] = timing_df_display['Green (s)'].apply(lambda x: f'🟢 {x:.1f}' if isinstance(x, (int, float)) else f'🟢 {x}')
    timing_df_display['Yellow (s)'] = timing_df_display['Yellow (s)'].apply(lambda x: f'🟡 {x:.1f}' if isinstance(x, (int, float)) else f'🟡 {x}')
    timing_df_display['Red (s)'] = timing_df_display['Red (s)'].apply(lambda x: f'🔴 {x:.1f}' if isinstance(x, (int, float)) else f'🔴 {x}')
    
    # Display the table
    st.dataframe(timing_df_display, use_container_width=True, hide_index=True)


# Define process_video function before it's used
def process_video(video_path, video_placeholder, timing_placeholder, 
                 stats_placeholder, model_path, conf_threshold, device,
                 num_lanes, roi_ratio, base_green, scaling_factor,
                 min_green, max_green, yellow_time, all_red_time,
                 target_fps, frame_skip, resize_video, use_weights,
                 weight_car, weight_motorcycle, weight_bus, weight_truck, 
                 weight_ambulance, weight_fire_truck, weight_police,
                 use_custom_model=False, emergency_class_map=None,
                 traffic_graph_placeholder=None, lane_graph_placeholder=None):
    """Process video and display results"""
    
    # Initialize components
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        # Load models
        status_text.text("Loading YOLOv8 model...")
        
        # Prepare vehicle classes and emergency class map
        vehicle_classes = None  # None = detect all (for custom models)
        if not use_custom_model:
            vehicle_classes = [2, 3, 5, 7]  # COCO: car, motorcycle, bus, truck
            emergency_class_map = None  # No emergency classes in COCO
        else:
            emergency_class_map = emergency_class_map or {}
        
        tracker = VehicleTracker(
            model_path=model_path,
            conf_threshold=conf_threshold,
            device=device,
            use_half=(device == "cuda"),
            tracker="bytetrack.yaml",
            vehicle_classes=vehicle_classes,
            emergency_class_map=emergency_class_map
        )
        
        # Initialize lane detector
        lane_detector = LaneDetector(num_lanes=num_lanes, roi_ratio=roi_ratio)
        
        # Initialize vehicle weight manager
        custom_weights = {
            "car": weight_car,
            "motorcycle": weight_motorcycle,
            "bus": weight_bus,
            "truck": weight_truck,
            "ambulance": weight_ambulance,
            "fire_truck": weight_fire_truck,
            "police": weight_police
        }
        weight_manager = VehicleWeightManager(custom_weights=custom_weights)
        
        # Initialize timing controller
        timing_controller = AdaptiveTimingController(
            base_green=base_green,
            scaling_factor=scaling_factor,
            min_green=min_green,
            max_green=max_green,
            yellow_time=yellow_time,
            all_red_time=all_red_time
        )
        
        # Initialize logger
        logger = TrafficLogger()
        st.session_state.logger = logger
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            st.error("Error opening video file")
            return
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        # Remove delay for maximum speed - process as fast as possible
        frame_delay = 0
        
        # Detect lanes from first few frames
        status_text.text("Detecting lanes...")
        lane_detected = False
        for i in range(min(10, total_frames)):
            ret, frame = cap.read()
            if not ret:
                break
            if resize_video:
                frame = resize_frame(frame, 640, 480)
            if lane_detector.detect_lanes(frame, auto_detect_lanes=True):
                lane_detected = True
                break
        
        if not lane_detected:
            # Fallback to simple partitioning
            test_width = 640 if resize_video else int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            lane_detector.frame_shape = (480 if resize_video else int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), test_width)
            lane_detector._simple_partition(test_width)
            status_text.text("Using default lane partitioning")
        
        # Initialize lane columns in logger
        lane_ids = list(range(len(lane_detector.get_lane_boundaries())))
        logger.add_lane_columns(lane_ids, include_weighted=use_weights)
        
        # Reset video capture
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        # Processing state is already set before calling this function
        # Just ensure stop flag is reset
        st.session_state.stop_processing = False
        
        # Processing loop
        frame_idx = 0
        processed_frames = 0
        
        status_text.text("Processing video...")
        
        while cap.isOpened():
            # Check for stop flag
            if st.session_state.stop_processing:
                status_text.warning("⏹️ Processing stopped by user")
                break
            
            ret, frame = cap.read()
            if not ret:
                break
            
            # Skip frames if needed
            if frame_idx % frame_skip != 0:
                frame_idx += 1
                continue
            
            # Resize if needed
            original_frame = frame.copy()
            if resize_video:
                frame = resize_frame(frame, 640, 480)
            
            # Track vehicles
            tracks = tracker.track(frame)
            
            # Detect emergency vehicles from class IDs (if using fine-tuned model)
            emergency_info = {}  # track_id -> emergency_type
            if use_custom_model and emergency_class_map:
                for track in tracks:
                    if track['id'] is not None:
                        class_id = track.get('class_id')
                        if class_id is not None:
                            emergency_type = tracker.is_emergency_vehicle(class_id)
                            if emergency_type:
                                emergency_info[track['id']] = emergency_type
            
            # Count vehicles per lane
            lane_vehicles = count_vehicles_per_lane(tracks, lane_detector)
            raw_counts = get_lane_counts(lane_vehicles)
            
            # Use weighted counts if enabled, otherwise use raw counts
            if use_weights:
                lane_counts = get_weighted_lane_counts(lane_vehicles, weight_manager, emergency_info)
            else:
                lane_counts = {lane_id: float(count) for lane_id, count in raw_counts.items()}
            
            # Ensure all lanes are in counts (even if empty)
            for lane_id in lane_ids:
                if lane_id not in lane_counts:
                    lane_counts[lane_id] = 0.0
            
            # Compute timings using weighted or raw counts
            current_time = processed_frames / fps if fps > 0 else processed_frames * 0.033
            timings = timing_controller.compute_timings(lane_counts, current_time)
            
            # Store raw counts in timings for display
            for lane_id in timings.keys():
                timings[lane_id]['raw_count'] = raw_counts.get(lane_id, 0)
                timings[lane_id]['weighted_count'] = lane_counts.get(lane_id, 0.0)
            
            # Log statistics (use raw_counts for logging)
            logger.log_frame(raw_counts, timings, fps)
            
            # Annotate frame (highlight emergency vehicles with different colors)
            annotated_frame = draw_vehicle_boxes(frame, tracks, draw_ids=False, 
                                                emergency_info=emergency_info)
            
            # Draw lane boundaries (optional, can be commented out)
            # annotated_frame = lane_detector.visualize_lanes(annotated_frame)
            
            # Convert BGR to RGB for Streamlit
            display_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            
            # Update traffic history for graphs
            update_traffic_history(lane_counts, current_time)
            
            # Update displays (update every few frames for speed, or every frame if target_fps is low)
            update_display = (frame_idx % max(1, frame_skip) == 0) or (target_fps < 10)
            
            if update_display:
                # Update displays
                video_placeholder.image(display_frame, channels="RGB", use_container_width=True)
                
                # Display timing table with icons below video
                with timing_placeholder.container():
                    display_timing_table_with_icons(timings)
                
                # Display statistics with better formatting
                total_raw = sum(raw_counts.values())
                total_weighted = sum(lane_counts.values()) if use_weights else total_raw
                with stats_placeholder.container():
                    st.markdown("### 📊 Statistics")
                    
                    # Metrics in columns
                    metric_cols = st.columns(3)
                    with metric_cols[0]:
                        st.metric("Total Vehicles", total_raw)
                    with metric_cols[1]:
                        st.metric("Weighted Count" if use_weights else "Total", 
                                 f"{total_weighted:.1f}" if use_weights else total_raw)
                    with metric_cols[2]:
                        st.metric("Active Tracks", len(tracker.get_active_track_ids()))
                    
                    # Additional stats
                    st.caption(f"Lanes: {len(lane_ids)} | FPS: {target_fps} | Time: {current_time:.1f}s")
                
                # Update real-time graphs (less frequently for speed)
                if frame_idx % 5 == 0:  # Update graphs every 5 frames
                    if traffic_graph_placeholder is not None:
                        with traffic_graph_placeholder.container():
                            create_traffic_graph()
                    
                    if lane_graph_placeholder is not None:
                        with lane_graph_placeholder.container():
                            create_lane_traffic_graph()
            
            # Update progress (every frame for smooth progress bar)
            progress = (frame_idx + 1) / total_frames
            progress_bar.progress(min(progress, 1.0))
            
            # Check for stop flag before continuing
            if st.session_state.stop_processing:
                break
            
            # No delay - process as fast as possible
            
            frame_idx += 1
            processed_frames += 1
        
        # Cleanup
        cap.release()
        progress_bar.empty()
        
        # Check if stopped before resetting state
        was_stopped = st.session_state.stop_processing
        
        # Reset processing state
        st.session_state.processing_active = False
        st.session_state.stop_processing = False
        
        if not was_stopped:
            status_text.success("✅ Video processing completed!")
            st.success(f"Processed {processed_frames} frames successfully")
        else:
            status_text.info("⏹️ Processing was stopped")
            st.info(f"Processed {processed_frames} frames before stopping")
        
    except Exception as e:
        st.session_state.processing_active = False
        st.session_state.stop_processing = False
        st.error(f"Error during processing: {str(e)}")
        import traceback
        st.code(traceback.format_exc())


# Sidebar for configuration
st.sidebar.header("⚙️ Configuration")

# Model selection
model_type = st.sidebar.radio(
    "Model Type",
    ["Pretrained COCO", "Custom Fine-Tuned"],
    help="Use pretrained COCO model or your custom fine-tuned model with emergency vehicles"
)

if model_type == "Pretrained COCO":
    model_size = st.sidebar.selectbox(
        "YOLOv8 Model Size",
        ["nano (yolov8n.pt)", "small (yolov8s.pt)", "medium (yolov8m.pt)"],
        index=1,  # Default to small for better accuracy
        help="nano=fastest/least accurate | small=balanced (recommended) | medium=most accurate/slower. Larger models detect motorcycles and small vehicles better."
    )
    model_path = model_size.split("(")[1].split(")")[0].strip()
    use_custom_model = False
else:
    model_path = st.sidebar.text_input(
        "Custom Model Path",
        value="best.pt",
        help="Path to your fine-tuned YOLOv8 model (e.g., 'best.pt' or 'runs/detect/train/weights/best.pt')"
    )
    use_custom_model = True
    
    # Emergency vehicle class IDs for custom model
    st.sidebar.subheader("Emergency Vehicle Class IDs")
    st.sidebar.caption("Enter class IDs from your fine-tuned model:")
    amb_class_id = st.sidebar.number_input("Ambulance Class ID", min_value=0, max_value=100, value=8, step=1,
                                           help="Class ID for ambulance in your custom model")
    fire_class_id = st.sidebar.number_input("Fire Truck Class ID", min_value=0, max_value=100, value=9, step=1,
                                            help="Class ID for fire truck in your custom model")
    police_class_id = st.sidebar.number_input("Police Car Class ID", min_value=0, max_value=100, value=10, step=1,
                                             help="Class ID for police car in your custom model")
    
    emergency_class_map = {
        'ambulance': int(amb_class_id),
        'fire_truck': int(fire_class_id),
        'police': int(police_class_id)
    }

# Detection settings
conf_threshold = st.sidebar.slider(
    "Confidence Threshold", 0.1, 0.9, 0.25, 0.05,
    help="Lower values detect more vehicles (including motorcycles) but may include false positives. Try 0.2-0.25 for better motorcycle detection."
)
device = st.sidebar.selectbox("Device", ["cuda", "cpu"], index=0 if os.environ.get("CUDA_VISIBLE_DEVICES") else 1)

# Lane detection settings
num_lanes = st.sidebar.number_input("Number of Lanes", min_value=1, max_value=6, value=2, step=1)
roi_ratio = st.sidebar.slider("ROI Ratio", 0.3, 0.9, 0.6, 0.1)

# Vehicle Weight Parameters
st.sidebar.subheader("Vehicle Weights")
use_weights = st.sidebar.checkbox("Enable Vehicle Weighting", value=True, 
                                  help="Weight vehicles by type (buses and trucks count more)")
enable_emergency_detection = st.sidebar.checkbox("Enable Emergency Vehicle Detection", value=True,
                                                 help="Detect and prioritize emergency vehicles")
weight_car = st.sidebar.number_input("Car Weight", min_value=1.0, max_value=10.0, value=3.0, step=0.5,
                                     help="Weight for cars")
weight_motorcycle = st.sidebar.number_input("Motorcycle/Bike Weight", min_value=1.0, max_value=10.0, value=2.0, step=0.5,
                                            help="Weight for motorcycles/bikes")
weight_bus = st.sidebar.number_input("Bus Weight", min_value=1.0, max_value=10.0, value=5.0, step=0.5,
                                     help="Weight for buses (carries more people)")
weight_truck = st.sidebar.number_input("Truck Weight", min_value=1.0, max_value=10.0, value=4.0, step=0.5,
                                       help="Weight for trucks")

# Emergency Vehicle Weights
st.sidebar.subheader("Emergency Vehicle Weights")
weight_ambulance = st.sidebar.number_input("Ambulance Weight", min_value=5.0, max_value=20.0, value=10.0, step=0.5,
                                           help="Weight for ambulances (highest priority)")
weight_fire_truck = st.sidebar.number_input("Fire Truck Weight", min_value=5.0, max_value=20.0, value=10.0, step=0.5,
                                            help="Weight for fire trucks (highest priority)")
weight_police = st.sidebar.number_input("Police Car Weight", min_value=5.0, max_value=20.0, value=8.0, step=0.5,
                                        help="Weight for police cars (very high priority)")

# Timing parameters
st.sidebar.subheader("Timing Parameters")
base_green = st.sidebar.number_input("Base Green Time (s)", min_value=3.0, max_value=20.0, value=5.0, step=0.5)
scaling_factor = st.sidebar.number_input("Scaling Factor (s/vehicle)", min_value=0.1, max_value=5.0, value=1.0, step=0.1)
min_green = st.sidebar.number_input("Min Green Time (s)", min_value=3.0, max_value=15.0, value=5.0, step=0.5)
max_green = st.sidebar.number_input("Max Green Time (s)", min_value=20.0, max_value=120.0, value=60.0, step=5.0)
yellow_time = st.sidebar.number_input("Yellow Time (s)", min_value=1.0, max_value=5.0, value=3.0, step=0.5)
all_red_time = st.sidebar.number_input("All-Red Time (s)", min_value=0.5, max_value=3.0, value=1.0, step=0.5)

# Performance settings
st.sidebar.subheader("Performance")
target_fps = st.sidebar.slider("Target FPS", 5, 60, 30, 1, 
                               help="Higher FPS = faster processing. Set lower if you want to see each frame clearly.")
frame_skip = st.sidebar.number_input("Process Every N Frames", min_value=1, max_value=5, value=1, step=1)
resize_video = st.sidebar.checkbox("Resize to 640x480", value=True)

# Main title
st.title("🚦 Smart Traffic Light Controller")
st.markdown("**YOLOv8-based adaptive traffic light control system**")

# Navigation
st.sidebar.markdown("---")
st.sidebar.markdown("### 🎯 Navigation")

# Mode selection
mode = st.sidebar.radio(
    "Operation Mode",
    ["Real Camera", "SUMO Simulation"],
    help="Choose between real camera video processing or SUMO simulation"
)

# SUMO Configuration (only show in SUMO mode)
sumo_config_file = None
show_sumo_gui = True
sumo_intersection_id = ""

if mode == "SUMO Simulation":
    st.sidebar.markdown("### 🚗 SUMO Configuration")
    sumo_config_file = st.sidebar.text_input(
        "SUMO Config File",
        value="sumo_scenarios/intersection/intersection.sumocfg",
        help="Path to SUMO configuration file (.sumocfg)"
    )
    show_sumo_gui = st.sidebar.checkbox("Show SUMO GUI", value=True, 
                                        help="Display SUMO graphical interface")
    sumo_intersection_id = st.sidebar.text_input(
        "Traffic Light ID",
        value="",
        help="Traffic light junction ID (leave empty to use first available)"
    )

# Initialize session state for SUMO
if 'sumo_sim' not in st.session_state:
    st.session_state.sumo_sim = None
    st.session_state.sumo_controller = None
    st.session_state.sumo_running = False

# SUMO Simulation Mode
if mode == "SUMO Simulation":
    if not SUMO_AVAILABLE:
        st.error("⚠️ SUMO integration not available. Please install: `pip install traci sumolib`")
        st.stop()
    
    if not sumo_config_file or not os.path.exists(sumo_config_file):
        st.warning("⚠️ Please provide a valid SUMO configuration file path")
        st.info("Example: Create a SUMO scenario and specify the path to the .sumocfg file")
        st.stop()
    
    # Create placeholder for timing display
    col1, col2 = st.columns([2, 1])
    
    with col1:
        sumo_status_placeholder = st.empty()
        st.subheader("🚗 SUMO Simulation")
    
    with col2:
        timing_placeholder = st.empty()
        stats_placeholder = st.empty()
        st.subheader("⏱️ Signal Timings")
    
    # Control buttons
    col_btn1, col_btn2, col_btn3, col_btn4 = st.columns(4)
    
    with col_btn1:
        if st.button("▶️ Start SUMO Simulation", type="primary"):
            try:
                # Initialize SUMO simulator
                sumo_sim = SUMOTrafficSimulator(
                    config_file=sumo_config_file,
                    gui=show_sumo_gui
                )
                
                # Start simulation
                traffic_light_ids = sumo_sim.start_simulation()
                
                if not traffic_light_ids:
                    st.warning("⚠️ No traffic lights detected by TraCI")
                    st.info("""
                    **Simulation will continue without traffic light control.**
                    
                    Vehicles will still move, but traffic lights won't be controlled adaptively.
                    
                    **To enable traffic light control:**
                    1. Signal plans need to be in correct format
                    2. Check that `intersection.add.xml` exists and has proper signal plans
                    3. The network may need to be reprocessed with netconvert
                    
                    **Current status:**
                    - Network: ✅ Loaded
                    - Routes: ✅ Loaded  
                    - Vehicles: ✅ Will spawn and move
                    - Traffic lights: ⚠️ Not detected (simulation continues anyway)
                    """)
                    # Continue anyway - simulation can run without traffic light control
                    # Use a dummy intersection ID for display purposes, but mark it as invalid
                    traffic_light_ids = ["A0"]  # Use first junction as placeholder
                    st.session_state.traffic_lights_available = False
                else:
                    st.session_state.traffic_lights_available = True
                
                # Use specified intersection ID or first available
                intersection_id = sumo_intersection_id if sumo_intersection_id else traffic_light_ids[0]
                
                if intersection_id not in traffic_light_ids:
                    st.warning(f"Traffic light '{intersection_id}' not found. Using '{traffic_light_ids[0]}'")
                    intersection_id = traffic_light_ids[0]
                
                # Initialize vehicle weight manager
                custom_weights = {
                    "car": weight_car,
                    "motorcycle": weight_motorcycle,
                    "bus": weight_bus,
                    "truck": weight_truck,
                    "ambulance": weight_ambulance,
                    "fire_truck": weight_fire_truck,
                    "police": weight_police
                }
                weight_manager = VehicleWeightManager(custom_weights=custom_weights)
                
                # Initialize timing controller
                timing_controller = AdaptiveTimingController(
                    base_green=base_green,
                    scaling_factor=scaling_factor,
                    min_green=min_green,
                    max_green=max_green,
                    yellow_time=yellow_time,
                    all_red_time=all_red_time
                )
                
                # Create SUMO traffic light controller
                sumo_controller = SUMOTrafficLightController(
                    sumo_sim=sumo_sim,
                    timing_controller=timing_controller,
                    weight_manager=weight_manager,
                    intersection_id=intersection_id
                )
                
                # Store in session state
                st.session_state.sumo_sim = sumo_sim
                st.session_state.sumo_controller = sumo_controller
                st.session_state.sumo_running = True
                st.session_state.intersection_id = intersection_id
                
                # Initialize logger
                if 'logger' not in st.session_state:
                    logger = TrafficLogger()
                    # Get lane IDs from controller - ensure we get all lanes
                    lane_counts = sumo_controller.get_vehicle_counts()
                    # Also get lane IDs from lane_mapping to ensure we have all lanes
                    lane_ids_from_mapping = list(sumo_controller.lane_mapping.values())
                    # Combine both sources and remove duplicates
                    all_lane_ids = sorted(set(list(lane_counts.keys()) + lane_ids_from_mapping))
                    if not all_lane_ids:
                        # Fallback: use 16 default lanes for 4-way intersection (4 per direction)
                        all_lane_ids = list(range(16))
                    logger.add_lane_columns(all_lane_ids, include_weighted=use_weights)
                    st.session_state.logger = logger
                
                st.success(f"✅ SUMO simulation started! Controlling traffic light: {intersection_id}")
                st.rerun()
                
            except Exception as e:
                st.error(f"Error starting SUMO simulation: {str(e)}")
                import traceback
                st.code(traceback.format_exc())
    
    with col_btn2:
        if st.button("⏹️ Stop Simulation", type="secondary"):
            if st.session_state.sumo_running and st.session_state.sumo_sim:
                try:
                    st.session_state.sumo_sim.close()
                    st.session_state.sumo_running = False
                    st.session_state.simulation_stopped = True
                    st.success("✅ Simulation stopped successfully!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error stopping simulation: {str(e)}")
                    st.session_state.sumo_running = False
                    st.session_state.simulation_stopped = True
            else:
                st.warning("No simulation running to stop.")
    
    with col_btn3:
        if st.button("🔄 Reset"):
            if st.session_state.sumo_sim:
                try:
                    st.session_state.sumo_sim.close()
                except:
                    pass
            st.session_state.sumo_sim = None
            st.session_state.sumo_controller = None
            st.session_state.sumo_running = False
            st.session_state.simulation_stopped = False
            if 'logger' in st.session_state:
                st.session_state.logger.reset()
            st.rerun()
    
    with col_btn4:
        if st.button("💾 Save Statistics"):
            if 'logger' in st.session_state:
                filename = st.session_state.logger.save_to_csv()
                st.success(f"Statistics saved to {filename}")
            else:
                st.warning("No statistics to save. Please start simulation first.")
    
    # Run simulation loop if active
    if st.session_state.sumo_running and st.session_state.sumo_sim and st.session_state.sumo_controller:
        # Check if traffic lights are available
        traffic_lights_available = st.session_state.get('traffic_lights_available', True)
        
        # Get vehicle counts from SUMO
        try:
            lane_counts = st.session_state.sumo_controller.get_vehicle_counts()
            
            # Get raw counts for display (works with or without traffic lights)
            raw_counts = {}
            try:
                lane_counts_raw = st.session_state.sumo_sim.get_vehicle_counts_per_lane(
                    st.session_state.intersection_id
                )
                for sumo_lane, count in lane_counts_raw.items():
                    internal_lane_id = st.session_state.sumo_controller.lane_mapping.get(sumo_lane, -1)
                    if internal_lane_id >= 0:
                        raw_counts[internal_lane_id] = count
            except:
                # Fallback: use counts from lane_counts if available
                raw_counts = {lid: int(count) for lid, count in lane_counts.items()}
        except Exception as e:
            # Handle any errors gracefully
            # Fallback: empty counts for up to 16 lanes
            raw_counts = {i: 0 for i in range(16)}
            lane_counts = {i: 0.0 for i in range(16)}
            print(f"Warning: Could not get vehicle counts: {e}")
        
        # Compute timings using existing controller logic
        current_time = st.session_state.sumo_sim.get_current_time()
        timings = st.session_state.sumo_controller.timing_controller.compute_timings(
            lane_counts, current_time
        )
        
        # Store raw counts in timings for display
        for lane_id in timings.keys():
            timings[lane_id]['raw_count'] = raw_counts.get(lane_id, 0)
            timings[lane_id]['weighted_count'] = lane_counts.get(lane_id, 0.0)
        
        # Log statistics
        if 'logger' in st.session_state:
            st.session_state.logger.log_frame(raw_counts, timings, fps=10.0)
        
        # Apply timings to SUMO using lane-based speed control (simulates traffic lights)
        # This works even without actual traffic lights in SUMO
        try:
            update_sumo_lights(st.session_state.sumo_controller, timings)
        except Exception as e:
            # Silently handle - lane-based control should work even if traffic lights don't exist
            pass
        
        # Step simulation (this now steps 10 times = 1 second)
        try:
            st.session_state.sumo_sim.step(1)
        except Exception as e:
            # If SUMO connection is lost, stop simulation and show results
            st.warning("⚠️ SUMO connection lost. Stopping simulation.")
            st.session_state.sumo_running = False
            st.session_state.simulation_stopped = True
            if st.session_state.sumo_sim:
                try:
                    st.session_state.sumo_sim.close()
                except:
                    pass
            st.rerun()
        
        # Format and display timing table
        timing_df = format_timing_table(timings)
        timing_placeholder.dataframe(timing_df, use_container_width=True, hide_index=True)
        
        # Display statistics
        total_raw = sum(raw_counts.values())
        total_weighted = sum(lane_counts.values()) if use_weights else total_raw
        stats_data = {
            'Metric': ['Total Vehicles (Raw)', 'Total Weighted Count' if use_weights else 'Total Vehicles',
                      'Simulation Time (s)', 'Traffic Light ID', 'Lanes'],
            'Value': [
                str(total_raw),
                f"{total_weighted:.1f}" if use_weights else str(total_raw),
                f"{current_time:.1f}",
                st.session_state.intersection_id,
                str(len(lane_counts))
            ]
        }
        stats_df = pd.DataFrame(stats_data)
        stats_placeholder.dataframe(stats_df, use_container_width=True, hide_index=True)
        
        # Display simulation status
        sumo_status_placeholder.info(
            f"🟢 Simulation Running | Time: {current_time:.1f}s | "
            f"Step: {st.session_state.sumo_sim.current_step} | "
            f"Vehicles: {total_raw}"
        )
        
        # Auto-refresh to continue simulation
        time.sleep(0.1)
        st.rerun()
    
    # Show results when simulation is stopped
    if st.session_state.get('simulation_stopped', False) and 'logger' in st.session_state:
        st.markdown("---")
        st.subheader("📊 Simulation Results")
        
        # Get logged data
        logger = st.session_state.logger
        if logger.log_data and len(logger.log_data.get('timestamp', [])) > 0:
            # Convert to DataFrame
            results_df = logger.to_dataframe()
            
            # Display summary statistics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                total_frames = len(results_df)
                st.metric("Total Frames", total_frames)
            
            with col2:
                if 'timestamp' in results_df.columns and len(results_df) > 0:
                    total_time = results_df['timestamp'].max()
                    st.metric("Total Time (s)", f"{total_time:.1f}")
                else:
                    st.metric("Total Time (s)", "0.0")
            
            with col3:
                # Calculate average vehicles per frame
                vehicle_cols = [col for col in results_df.columns if 'lane_' in col and '_count' in col and 'weighted' not in col]
                if vehicle_cols:
                    avg_vehicles = results_df[vehicle_cols].sum(axis=1).mean()
                    st.metric("Avg Vehicles/Frame", f"{avg_vehicles:.1f}")
                else:
                    st.metric("Avg Vehicles/Frame", "0")
            
            with col4:
                # Calculate total vehicles processed
                if vehicle_cols:
                    total_vehicles = results_df[vehicle_cols].sum(axis=1).sum()
                    st.metric("Total Vehicles", int(total_vehicles))
                else:
                    st.metric("Total Vehicles", "0")
            
            # Display detailed statistics table
            st.markdown("### 📈 Detailed Statistics")
            st.dataframe(results_df, use_container_width=True, height=400)
            
            # Display lane-wise summary
            st.markdown("### 🛣️ Lane-wise Summary")
            lane_summary = []
            # Map lane IDs to directions for 4-way intersection
            direction_names = {0: "North", 1: "East", 2: "South", 3: "West"}
            for col in vehicle_cols:
                lane_id = col.replace('lane_', '').replace('_count', '')
                avg_count = results_df[col].mean()
                max_count = results_df[col].max()
                total_count = results_df[col].sum()
                lane_name = direction_names.get(int(lane_id), f"Lane {int(lane_id) + 1}")
                
                lane_summary.append({
                    'Lane': lane_name,
                    'Avg Vehicles': f"{avg_count:.2f}",
                    'Max Vehicles': int(max_count),
                    'Total Vehicles': int(total_count)
                })
            
            if lane_summary:
                summary_df = pd.DataFrame(lane_summary)
                st.dataframe(summary_df, use_container_width=True, hide_index=True)
            
            # Download button for results
            csv_data = results_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Results as CSV",
                data=csv_data,
                file_name=f"simulation_results_{int(time.time())}.csv",
                mime="text/csv"
            )
        else:
            st.info("No data collected during simulation.")

# Real Camera Mode
if mode == "Real Camera":
    # File uploader
    uploaded_file = st.file_uploader(
        "Upload Traffic Video (MP4)",
        type=['mp4', 'avi', 'mov'],
        help="Upload a video file showing traffic at an intersection"
    )
    
    # Video processing section
    if uploaded_file is not None:
        # Save uploaded file temporarily
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        tfile.write(uploaded_file.read())
        video_path = tfile.name
        tfile.close()
        
        st.session_state.video_file = video_path
        
        # Create placeholder for video, timing, and graphs
        # Video takes full width
        video_placeholder = st.empty()
        st.subheader("📹 Processed Video")
        
        # Signal Timings below video (full width)
        timing_placeholder = st.empty()
        
        # Statistics and graphs in columns
        stats_col1, stats_col2 = st.columns(2)
        with stats_col1:
            stats_placeholder = st.empty()
        
        with stats_col2:
            st.markdown("### 📈 Real-Time Traffic Analytics")
            graph_col1, graph_col2 = st.columns(2)
            with graph_col1:
                st.markdown("#### Total Vehicle Count")
                traffic_graph_placeholder = st.empty()
            
            with graph_col2:
                st.markdown("#### Lane-wise Traffic")
                lane_graph_placeholder = st.empty()
        
        # Control buttons
        col_btn1, col_btn2, col_btn3 = st.columns(3)
        
        with col_btn1:
            # Show different button based on processing state
            if st.session_state.get('processing_active', False):
                if st.button("⏹️ Stop Processing", type="primary"):
                    st.session_state.stop_processing = True
                    st.session_state.processing_active = False
                    st.rerun()
            else:
                if st.button("▶️ Start Processing", type="primary"):
                    # Set processing state immediately so button changes on next rerun
                    st.session_state.processing_active = True
                    st.session_state.stop_processing = False
                    st.rerun()  # Rerun to update button to "Stop Processing" immediately
        
        # Process video if processing is active and not stopped
        if st.session_state.get('processing_active', False) and not st.session_state.get('stop_processing', False):
            emergency_map = emergency_class_map if use_custom_model else None
            process_video(video_path, video_placeholder, timing_placeholder, 
                         stats_placeholder, model_path, conf_threshold, device,
                         num_lanes, roi_ratio, base_green, scaling_factor,
                         min_green, max_green, yellow_time, all_red_time,
                         target_fps, frame_skip, resize_video, use_weights,
                         weight_car, weight_motorcycle, weight_bus, weight_truck,
                         weight_ambulance, weight_fire_truck, weight_police,
                         use_custom_model, emergency_map,
                         traffic_graph_placeholder, lane_graph_placeholder)
        
        with col_btn2:
            if st.button("🔄 Reset"):
                st.session_state.video_processed = False
                st.session_state.processing_active = False
                st.session_state.stop_processing = False
                st.rerun()
        
        with col_btn3:
            if st.button("💾 Save Statistics"):
                if 'logger' in st.session_state:
                    filename = st.session_state.logger.save_to_csv()
                    st.success(f"Statistics saved to {filename}")
                else:
                    st.warning("No statistics to save. Please process video first.")
        
        # Display video info
        cap = cv2.VideoCapture(video_path)
        if cap.isOpened():
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            duration = frame_count / fps if fps > 0 else 0
            
            cap.release()
            
            st.info(f"📊 Video Info: {width}x{height} @ {fps:.2f} FPS, {duration:.1f}s duration, {frame_count} frames")

if __name__ == "__main__":
    if mode == "Real Camera":
        st.info("👆 Upload a video file to get started!")
    else:
        st.info("👆 Configure SUMO settings and click 'Start SUMO Simulation' to begin!")

