# 🚦 Traffic Violation Detection System

A comprehensive real-time traffic violation detection system using computer vision and deep learning to monitor and detect various traffic violations with automated image capture and web dashboard.

## ✨ Features

### 🎯 Violation Detection Types
- **Red Light Violations** - Detects vehicles crossing stop lines during red lights
- **Wrong Side Driving** - Identifies vehicles driving in opposite direction
- **Heavy Vehicle Prohibition** - Monitors restricted zones for heavy vehicles
- **Parking Violations** - Detects unauthorized parking with timer-based tracking
- **Lane Change Violations** - Identifies illegal lane switching behavior

### 🔧 Core Capabilities
- **Real-time Video Processing** - Live traffic monitoring with YOLO object detection
- **Interactive Zone Drawing** - Easy-to-use zone configuration tool
- **Multi-object Tracking** - DeepSORT-based vehicle tracking
- **Automated Image Capture** - Date-organized violation evidence storage
- **Web Dashboard** - Real-time streaming with zone management interface
- **Headless Operation** - Server-compatible deployment without GUI requirements

## 🚀 Quick Start

### Prerequisites
```bash
pip install opencv-python ultralytics flask deep-sort-realtime numpy
```

### 1. Draw Detection Zones
```bash
python simple_zone_drawer.py
```
- Load your video file
- Draw zones for different violation types
- Save configuration automatically

### 2. Run Detection System
```bash
python main_app.py --video dataset/your_video.mp4
```

### 3. Access Web Dashboard
Open `http://localhost:5000` in your browser for real-time monitoring

## 📁 Project Structure

```
Manus_traffic/
├── main_app.py                    # Main Flask application
├── simple_zone_drawer.py          # Interactive zone drawing tool
├── violation_detectors.py         # Violation detection algorithms
├── traffic_violation_detector.py  # Core detection components
├── config.json                    # Zone configuration
├── violations/                    # Date-organized violation images
│   └── YYYY-MM-DD/
│       ├── red_light_violation/
│       ├── parking_violation/
│       └── ...
└── dataset/                       # Video files
```

## 🎮 Usage

### Zone Drawing Tool
```bash
# Interactive zone configuration
python simple_zone_drawer.py

# Controls:
# - Left click: Add zone points
# - Right click: Complete zone
# - 'r': Reset current zone
# - 'c': Clear all zones
# - 's': Save configuration
# - 'q': Quit
```

### Detection System
```bash
# Basic usage
python main_app.py --video dataset/traffic.mp4

# Headless mode (no display)
python main_app.py --video dataset/traffic.mp4 --no-display

# Custom configuration
python main_app.py --video dataset/traffic.mp4 --config custom_config.json
```

## 🔍 Detection Algorithms

### Red Light Detection
- **HSV Color Analysis** - Precise red color detection in traffic lights
- **BGR Validation** - Secondary color verification
- **Brightness Checking** - Light intensity analysis
- **Zone-specific Triggers** - Only vehicles crossing defined stop lines

### Parking Violation
- **Position-based Tracking** - 20px grid cell system for accurate positioning
- **Timer Management** - Configurable violation thresholds
- **Continuous Monitoring** - Regular violation image capture every 5 seconds

### Lane Change Detection
- **Trajectory Analysis** - Vehicle path monitoring
- **Lane Boundary Recognition** - Automated lane detection
- **Illegal Movement Identification** - Unsafe lane switching detection

## 🌐 Web Dashboard

Access the web interface at `http://localhost:5000`:

- **Live Video Stream** - Real-time traffic monitoring
- **Zone Management** - Visual zone configuration
- **Violation Gallery** - Browse captured violations by date
- **System Status** - Monitor detection performance

## ⚙️ Configuration

### Zone Types
1. **Stop Line** - Red light violation detection
2. **No Parking** - Parking violation monitoring
3. **Heavy Vehicle Prohibited** - Weight restriction enforcement
4. **Wrong Side** - Direction violation detection
5. **Lane Change** - Lane switching monitoring
6. **Speed Limit** - Speed violation detection

### Video Requirements
- Supported formats: MP4, AVI, MOV, MKV
- Recommended resolution: 720p or higher
- Place videos in `dataset/` folder

## 📊 Output

### Violation Images
- Automatically saved with timestamps
- Organized by date and violation type
- High-quality evidence capture
- Bounding box annotations

### File Organization
```
violations/
└── 2024-01-15/
    ├── red_light_violation/
    │   ├── violation_20240115_143022.jpg
    │   └── violation_20240115_143045.jpg
    ├── parking_violation/
    └── lane_change_violation/
```

## 🛠️ Technical Details

- **Object Detection**: YOLOv8 for vehicle recognition
- **Tracking**: DeepSORT for multi-object tracking
- **Backend**: Flask web framework
- **Computer Vision**: OpenCV for image processing
- **Storage**: Date-based file organization

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/new-detection`)
3. Commit changes (`git commit -am 'Add new detection type'`)
4. Push to branch (`git push origin feature/new-detection`)
5. Create Pull Request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🔧 Troubleshooting

### Common Issues
- **OpenCV GUI Error**: Use `--no-display` flag for headless operation
- **Video Not Loading**: Ensure video path includes `dataset/` folder
- **Zone Configuration**: Use `simple_zone_drawer.py` before running detection
- **Missing Dependencies**: Install all required packages from prerequisites

### Performance Tips
- Use GPU acceleration if available
- Optimize video resolution for better performance
- Configure detection zones precisely for accuracy

---

**Built with ❤️ for traffic safety and monitoring**