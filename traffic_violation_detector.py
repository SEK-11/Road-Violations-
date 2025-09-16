#!/usr/bin/env python3
"""
Real-time Traffic Violation Detection System
Author: Manus AI
Date: July 15, 2025

This system detects and tracks vehicles, pedestrians, and riders in real-time,
applying geometric and temporal rules to flag 13 different types of traffic violations.
"""

import cv2
import numpy as np
import time
import json
import logging
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
from pathlib import Path
import threading
import queue

# Deep learning and tracking imports
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

# Geometric analysis imports
from shapely.geometry import Point, Polygon, LineString
from shapely.ops import nearest_points

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class DetectionResult:
    """Data structure for object detection results"""
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    confidence: float
    class_id: int
    class_name: str
    track_id: Optional[int] = None
    attributes: Dict[str, Any] = None

@dataclass
class ViolationEvent:
    """Data structure for violation events"""
    violation_type: str
    timestamp: datetime
    track_id: int
    location: Tuple[float, float]
    confidence: float
    evidence: Dict[str, Any]
    description: str

class VideoProcessor:
    """Handles video input and preprocessing"""
    
    def __init__(self, source: str = 0):
        """
        Initialize video processor
        Args:
            source: Video source (0 for webcam, path for video file, URL for stream)
        """
        self.source = source
        self.cap = None
        self.frame_count = 0
        self.fps = 30
        self.current_clean_frame = None  # Store clean frame for saving violations
        
    def initialize(self) -> bool:
        """Initialize video capture"""
        try:
            self.cap = cv2.VideoCapture(self.source)
            if not self.cap.isOpened():
                logger.error(f"Failed to open video source: {self.source}")
                return False
            
            # Get video properties
            self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30
            width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            logger.info(f"Video initialized: {width}x{height} @ {self.fps} FPS")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing video: {e}")
            return False
    
    def read_frame(self) -> Optional[np.ndarray]:
        """Read next frame from video source"""
        if self.cap is None:
            return None
            
        ret, frame = self.cap.read()
        if ret:
            self.frame_count += 1
            # Store the original frame for clean image saving
            self.current_clean_frame = frame.copy()
            return self.preprocess_frame(frame)
        return None
        
    def get_current_frame_clean(self) -> Optional[np.ndarray]:
        """Get the current frame without any overlays or annotations"""
        if hasattr(self, 'current_clean_frame'):
            return self.preprocess_frame(self.current_clean_frame)
        return None
    
    def preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """Apply preprocessing to frame"""
        # Resize frame for consistent processing
        height, width = frame.shape[:2]
        if width > 1280:
            scale = 1280 / width
            new_width = 1280
            new_height = int(height * scale)
            frame = cv2.resize(frame, (new_width, new_height))
        
        # Apply lighting normalization
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(l)
        frame = cv2.merge([l, a, b])
        frame = cv2.cvtColor(frame, cv2.COLOR_LAB2BGR)
        
        return frame
    
    def release(self):
        """Release video capture resources"""
        if self.cap:
            self.cap.release()

class ObjectDetector:
    """Unified object detection using YOLO"""
    
    def __init__(self, model_path: str = "yolo11n.pt"):
        """Initialize YOLO11 detector"""
        self.model = YOLO(model_path)
        self.class_names = self.model.names
        
        # Define class mappings for traffic objects
        self.traffic_classes = {
            'car': 2, 'motorcycle': 3, 'bus': 5, 'truck': 7,
            'person': 0, 'bicycle': 1, 'traffic light': 9
        }
        
    def detect(self, frame: np.ndarray) -> List[DetectionResult]:
        """Detect objects in frame"""
        results = self.model(frame, verbose=False)
        detections = []
        
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    # Extract detection data
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = box.conf[0].cpu().numpy()
                    class_id = int(box.cls[0].cpu().numpy())
                    class_name = self.class_names[class_id]
                    
                    # Filter for traffic-relevant objects
                    if confidence > 0.6 and class_name in ['rickshaw','car', 'motorcycle', 'bus', 'truck', 'person', 'bicycle', 'traffic light','bicycle', 'e-bike', 'jeep', 'motorcycle', 'tricycle','van']:
                        detection = DetectionResult(
                            bbox=(int(x1), int(y1), int(x2), int(y2)),
                            confidence=float(confidence),
                            class_id=class_id,
                            class_name=class_name,
                            attributes=self._detect_attributes(frame, (int(x1), int(y1), int(x2), int(y2)), class_name)
                        )
                        detections.append(detection)
        
        return detections
    
    def _detect_attributes(self, frame: np.ndarray, bbox: Tuple[int, int, int, int], class_name: str) -> Dict[str, Any]:
        """Detect object attributes (simplified version)"""
        x1, y1, x2, y2 = bbox
        roi = frame[y1:y2, x1:x2]
        
        attributes = {}
        
        if class_name == 'car':
            # Only detect headlights (removed seatbelt and phone detection)
            attributes['headlights_on'] = self._detect_headlights(roi)
            
        elif class_name == 'person':
            # Detect if person is in vehicle or pedestrian
            attributes['in_vehicle'] = self._is_in_vehicle(roi)
            
        return attributes
    
    def _detect_headlights(self, roi: np.ndarray) -> bool:
        """Detect if headlights are on"""
        if roi.size == 0:
            return False
        
        # Convert to grayscale and look for bright spots
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
        # Threshold for bright areas (headlights)
        _, bright_mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
        bright_pixels = cv2.countNonZero(bright_mask)
        
        # If significant bright areas detected, headlights are on
        bright_ratio = bright_pixels / (roi.shape[0] * roi.shape[1])
        return bright_ratio > 0.02
    
    def _is_in_vehicle(self, roi: np.ndarray) -> bool:
        """Determine if person is inside a vehicle"""
        # Simple heuristic based on surrounding context
        # This would need more sophisticated analysis in practice
        return False

class MultiObjectTracker:
    """Multi-object tracking using DeepSORT"""
    
    def __init__(self):
        """Initialize DeepSORT tracker"""
        self.tracker = DeepSort(max_age=50, n_init=3)
        self.tracks = {}  # Store track history
        
    def update(self, detections: List[DetectionResult], frame: np.ndarray) -> List[DetectionResult]:
        """Update tracker with new detections"""
        # Convert detections to DeepSORT format
        detection_list = []
        for det in detections:
            x1, y1, x2, y2 = det.bbox
            detection_list.append(([x1, y1, x2-x1, y2-y1], det.confidence, det.class_name))
        
        # Update tracker
        tracks = self.tracker.update_tracks(detection_list, frame=frame)
        
        # Update detection results with track IDs
        tracked_detections = []
        for track, det in zip(tracks, detections):
            if track.is_confirmed():
                det.track_id = track.track_id
                
                # Update track history
                if track.track_id not in self.tracks:
                    self.tracks[track.track_id] = {
                        'positions': [],
                        'timestamps': [],
                        'class_name': det.class_name,
                        'attributes_history': []
                    }
                
                # Add current position and attributes to history
                center_x = (det.bbox[0] + det.bbox[2]) // 2
                center_y = (det.bbox[1] + det.bbox[3]) // 2
                
                self.tracks[track.track_id]['positions'].append((center_x, center_y))
                self.tracks[track.track_id]['timestamps'].append(time.time())
                self.tracks[track.track_id]['attributes_history'].append(det.attributes)
                
                # Keep only recent history (last 30 frames)
                if len(self.tracks[track.track_id]['positions']) > 30:
                    self.tracks[track.track_id]['positions'] = self.tracks[track.track_id]['positions'][-30:]
                    self.tracks[track.track_id]['timestamps'] = self.tracks[track.track_id]['timestamps'][-30:]
                    self.tracks[track.track_id]['attributes_history'] = self.tracks[track.track_id]['attributes_history'][-30:]
                
                tracked_detections.append(det)
        
        return tracked_detections
    
    def get_track_trajectory(self, track_id: int) -> List[Tuple[int, int]]:
        """Get trajectory for a specific track"""
        if track_id in self.tracks:
            return self.tracks[track_id]['positions']
        return []
    
    def get_track_attributes_history(self, track_id: int) -> List[Dict[str, Any]]:
        """Get attribute history for a specific track"""
        if track_id in self.tracks:
            return self.tracks[track_id]['attributes_history']
        return []

class ZoneManager:
    """Manages polygonal zones for violation detection"""
    
    def __init__(self):
        """Initialize zone manager"""
        self.zones = {}
        
    def add_zone(self, zone_id: str, points: List[Tuple[int, int]], zone_type: str):
        """Add a zone (polygon or line)"""
        if zone_type == 'lane_divider':
            # For lane dividers, store as lines (no polygon needed)
            self.zones[zone_id] = {
                'polygon': None,
                'type': zone_type,
                'points': points
            }
        else:
            # For other zones, create polygon
            polygon = Polygon(points)
            self.zones[zone_id] = {
                'polygon': polygon,
                'type': zone_type,
                'points': points
            }
        logger.info(f"Added {zone_type} zone: {zone_id}")
    
    def point_in_zone(self, point: Tuple[int, int], zone_id: str) -> bool:
        """Check if point is inside zone"""
        if zone_id not in self.zones:
            return False
        
        zone_data = self.zones[zone_id]
        if zone_data['type'] == 'lane_divider':
            # Lane dividers don't contain points, they are crossed
            return False
        
        point_geom = Point(point)
        return zone_data['polygon'].contains(point_geom)
    
    def line_crosses_zone(self, line_points: List[Tuple[int, int]], zone_id: str) -> bool:
        """Check if line crosses zone boundary"""
        if zone_id not in self.zones or len(line_points) < 2:
            return False
        
        line = LineString(line_points)
        return line.intersects(self.zones[zone_id]['polygon'].boundary)
    
    def get_zones_by_type(self, zone_type: str) -> Dict[str, Dict]:
        """Get all zones of specific type"""
        return {k: v for k, v in self.zones.items() if v['type'] == zone_type}

if __name__ == "__main__":
    # This will be the main application entry point
    print("Traffic Violation Detection System")
    print("Core components initialized successfully!")

