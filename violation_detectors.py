#!/usr/bin/env python3
"""
Violation Detection Modules
Author: Manus AI
Date: July 15, 2025

This module contains all the specific violation detection algorithms.
"""

import cv2
import numpy as np
import time
import os
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import logging
from pathlib import Path

from traffic_violation_detector import DetectionResult, ViolationEvent, ZoneManager
from shapely.geometry import Point

logger = logging.getLogger(__name__)

class BaseViolationDetector:
    """Base class for all violation detectors"""
    
    def __init__(self, name: str):
        self.name = name
        self.enabled = True
        self.violation_count = 0
        
    def detect(self, detections: List[DetectionResult], frame: np.ndarray, 
               tracker_data: Dict, zones: ZoneManager) -> List[ViolationEvent]:
        """Override this method in subclasses"""
        raise NotImplementedError

class RedLightViolationDetector(BaseViolationDetector):
    """Detects red light and stop line violations"""
    
    def __init__(self):
        super().__init__("Red Light Violation")
        self.traffic_light_state = "unknown"
        self.state_change_time = time.time()
        self.vehicles_in_intersection = set()
        self.processed_violations = set()  # Track IDs of vehicles already processed
        
    def detect(self, detections: List[DetectionResult], frame: np.ndarray, 
               tracker_data: Dict, zones: ZoneManager) -> List[ViolationEvent]:
        violations = []
        
        # Get stop line zones
        stop_line_zones = zones.get_zones_by_type("stop_line")
        
        # Only continue if stop line zones are defined
        if not stop_line_zones:
            return violations
        
        # Find traffic lights in the frame
        traffic_lights = [d for d in detections if d.class_name == 'traffic light']
        
        # Only process if traffic lights are detected
        if traffic_lights:
            # Update traffic light state based on detected traffic lights
            self._update_traffic_light_state(traffic_lights, frame)
        else:
            # No traffic lights detected, set state to unknown
            if self.traffic_light_state != "unknown":
                self.traffic_light_state = "unknown"
                logger.info("No traffic light detected, setting state to unknown")
        
        # Draw traffic light state on frame
        self._draw_traffic_light_state(frame)
        
        # Only detect violations if traffic lights are present and red
        if traffic_lights and self.traffic_light_state == "red":
            for detection in detections:
                if detection.class_name in ['car', 'motorcycle', 'bus', 'truck'] and detection.track_id:
                    # Check if vehicle crosses stop line after red light
                    if self._check_stop_line_violation(detection, tracker_data, stop_line_zones):
                        # Skip if we've already processed this vehicle
                        if detection.track_id in self.processed_violations:
                            continue
                            
                        # Create violation event
                        violation = ViolationEvent(
                            violation_type="red_light_violation",
                            timestamp=datetime.now(),
                            track_id=detection.track_id,
                            location=(detection.bbox[0], detection.bbox[1]),
                            confidence=0.9,
                            evidence={
                                "traffic_light_state": self.traffic_light_state,
                                "bbox": detection.bbox,
                                "vehicle_type": detection.class_name
                            },
                            description=f"{detection.class_name} crossed stop line during red light"
                        )
                        violations.append(violation)
                        self.violation_count += 1
                        
                        # Mark as processed
                        self.processed_violations.add(detection.track_id)
                        
                        logger.warning(f"Red light violation detected: Track {detection.track_id}")
        else:
            # If light is not red, allow reprocessing of vehicles
            self.processed_violations.clear()
        
        return violations
        
    # Image saving method removed - now handled by the main application
    
    def _draw_traffic_light_state(self, frame: np.ndarray):
        """Draw the current traffic light state on the frame"""
        # Define colors for different states
        colors = {
            "red": (0, 0, 255),
            "yellow": (0, 255, 255),
            "green": (0, 255, 0),
            "unknown": (128, 128, 128)
        }
        
        color = colors.get(self.traffic_light_state, colors["unknown"])
        
        # Draw a traffic light indicator in the top-right corner
        height, width = frame.shape[:2]
        indicator_size = 30
        margin = 10
        
        # Draw background
        cv2.rectangle(frame, 
                     (width - indicator_size - margin - 150, margin), 
                     (width - margin, margin + indicator_size + 40), 
                     (0, 0, 0), -1)
        
        # Draw circle for current state
        cv2.circle(frame, 
                  (width - indicator_size//2 - margin - 50, margin + indicator_size//2), 
                  indicator_size//2, color, -1)
        
        # Add text with more information
        if self.traffic_light_state == "unknown":
            status_text = "No Traffic Light Detected"
        else:
            status_text = f"Light: {self.traffic_light_state.upper()}"
            
        cv2.putText(frame, status_text, 
                   (width - indicator_size - margin - 145, margin + indicator_size + 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    def _update_traffic_light_state(self, traffic_lights: List[DetectionResult], frame: np.ndarray):
        """Update traffic light state based on detected traffic lights"""
        for detection in traffic_lights:
            # Extract traffic light region
            x1, y1, x2, y2 = detection.bbox
            light_roi = frame[y1:y2, x1:x2]
            
            if light_roi.size > 0:
                # Analyze colors to determine state
                state = self._analyze_traffic_light_colors(light_roi)
                
                # Draw bounding box around traffic light with color indicating state
                color = (0, 0, 255) if state == "red" else (0, 255, 255) if state == "yellow" else (0, 255, 0) if state == "green" else (255, 255, 255)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"Traffic Light: {state.upper()}", (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                # Update state if it changed
                if state != self.traffic_light_state:
                    self.traffic_light_state = state
                    self.state_change_time = time.time()
                    logger.info(f"Traffic light state changed to: {state}")
                break
    
    def _detect_traffic_lights_by_color(self, frame: np.ndarray):
        """Try to detect traffic lights by color analysis when YOLO doesn't detect them"""
        # Disable this feature - we should only detect traffic lights through YOLO
        # If no traffic light is detected, set state to unknown
        if self.traffic_light_state != "unknown":
            self.traffic_light_state = "unknown"
            logger.info("No traffic light detected, setting state to unknown")
        return
    
    def _analyze_traffic_light_colors(self, light_roi: np.ndarray) -> str:
        """Analyze traffic light colors to determine state with improved accuracy"""
        if light_roi.size == 0:
            return "unknown"
            
        # Method 1: HSV color analysis (original)
        hsv = cv2.cvtColor(light_roi, cv2.COLOR_BGR2HSV)
        
        # Improved HSV ranges for better accuracy
        red_lower1 = np.array([0, 120, 70])     # More restrictive saturation/value
        red_upper1 = np.array([10, 255, 255])
        red_lower2 = np.array([170, 120, 70])   # Adjusted for better red detection
        red_upper2 = np.array([180, 255, 255])
        
        yellow_lower = np.array([15, 150, 150])  # More restrictive for yellow
        yellow_upper = np.array([35, 255, 255])
        
        green_lower = np.array([40, 120, 70])    # More restrictive for green
        green_upper = np.array([80, 255, 255])
        
        # Create masks
        red_mask1 = cv2.inRange(hsv, red_lower1, red_upper1)
        red_mask2 = cv2.inRange(hsv, red_lower2, red_upper2)
        red_mask = cv2.bitwise_or(red_mask1, red_mask2)
        
        yellow_mask = cv2.inRange(hsv, yellow_lower, yellow_upper)
        green_mask = cv2.inRange(hsv, green_lower, green_upper)
        
        # Count pixels
        red_pixels = cv2.countNonZero(red_mask)
        yellow_pixels = cv2.countNonZero(yellow_mask)
        green_pixels = cv2.countNonZero(green_mask)
        
        # Method 2: BGR color analysis for additional validation
        bgr_red_pixels = self._count_bgr_red_pixels(light_roi)
        
        # Method 3: Brightness analysis (red lights are often brightest)
        gray = cv2.cvtColor(light_roi, cv2.COLOR_BGR2GRAY)
        bright_pixels = cv2.countNonZero(cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)[1])
        
        # Calculate total pixels for percentage
        total_pixels = light_roi.shape[0] * light_roi.shape[1]
        
        # Improved decision logic
        red_score = red_pixels + (bgr_red_pixels * 0.5)
        yellow_score = yellow_pixels
        green_score = green_pixels
        
        # Minimum threshold - need at least 5% colored pixels
        min_threshold = total_pixels * 0.05
        
        if red_score < min_threshold and yellow_score < min_threshold and green_score < min_threshold:
            return "unknown"
        
        # Red detection with additional validation
        if red_score >= max(yellow_score, green_score):
            # Additional validation for red: check if it's also bright
            if bright_pixels > total_pixels * 0.1:  # At least 10% bright pixels
                return "red"
            elif red_score > min_threshold * 2:  # Strong red signal
                return "red"
        
        # Yellow and green detection
        if yellow_score >= max(red_score, green_score) and yellow_score > min_threshold:
            return "yellow"
        elif green_score >= max(red_score, yellow_score) and green_score > min_threshold:
            return "green"
        
        return "unknown"
    
    def _count_bgr_red_pixels(self, roi: np.ndarray) -> int:
        """Count red pixels using BGR color space for additional validation"""
        # In BGR, red pixels have high B component and low G component
        b, g, r = cv2.split(roi)
        
        # Red condition: R > 100, R > G+30, R > B+30
        red_condition = (r > 100) & (r > g + 30) & (r > b + 30)
        return np.sum(red_condition)
    
    def _check_stop_line_violation(self, detection: DetectionResult, tracker_data: Dict, 
                                 stop_line_zones: Dict) -> bool:
        """Check if vehicle violated stop line - only vehicles crossing the specific zone"""
        if not detection.track_id or detection.track_id not in tracker_data:
            return False
        
        track_positions = tracker_data[detection.track_id]['positions']
        if len(track_positions) < 2:
            return False
        
        # Get vehicle center point
        x1, y1, x2, y2 = detection.bbox
        vehicle_center = ((x1 + x2) // 2, (y1 + y2) // 2)
        vehicle_point = Point(vehicle_center[0], vehicle_center[1])
        
        # Only check vehicles that are currently in or crossing the specific stop line zone
        for zone_id, zone_data in stop_line_zones.items():
            zone_polygon = zone_data['polygon']
            
            # Check if vehicle is currently in the stop line zone
            if zone_polygon.contains(vehicle_point):
                # Check trajectory to see if it entered the zone during red light
                recent_positions = track_positions[-5:]  # Last 5 positions
                
                for i in range(len(recent_positions) - 1):
                    p1 = Point(recent_positions[i][0], recent_positions[i][1])
                    p2 = Point(recent_positions[i+1][0], recent_positions[i+1][1])
                    
                    # If vehicle crossed into the zone (was outside, now inside)
                    if not zone_polygon.contains(p1) and zone_polygon.contains(p2):
                        # Check if it was after red light with grace period
                        if time.time() - self.state_change_time > 1.0:  # Grace period
                            return True
        
        return False

class WrongSideDrivingDetector(BaseViolationDetector):
    """Detects wrong side driving violations"""
    
    def __init__(self):
        super().__init__("Wrong Side Driving")
        
    def detect(self, detections: List[DetectionResult], frame: np.ndarray, 
               tracker_data: Dict, zones: ZoneManager) -> List[ViolationEvent]:
        violations = []
        
        # Get lane zones
        lane_zones = zones.get_zones_by_type("lane")
        
        # Only detect violations if lane zones are defined
        if not lane_zones:
            return violations
        
        # Draw lane directions on frame for visualization
        for zone_id, zone_data in lane_zones.items():
            points = zone_data['points']
            if len(points) >= 2:
                # Determine expected direction from zone ID
                if "_left" in zone_id.lower():
                    direction_text = "← LEFT ONLY"
                    arrow_start = (points[0][0] + 100, (points[0][1] + points[-1][1]) // 2)
                    arrow_end = (points[0][0] + 20, (points[0][1] + points[-1][1]) // 2)
                elif "_right" in zone_id.lower():
                    direction_text = "RIGHT ONLY →"
                    arrow_start = (points[0][0] + 20, (points[0][1] + points[-1][1]) // 2)
                    arrow_end = (points[0][0] + 100, (points[0][1] + points[-1][1]) // 2)
                else:
                    direction_text = "RIGHT ONLY →"  # Default
                    arrow_start = (points[0][0] + 20, (points[0][1] + points[-1][1]) // 2)
                    arrow_end = (points[0][0] + 100, (points[0][1] + points[-1][1]) // 2)
                
                # Draw lane polygon
                pts = np.array(points, np.int32)
                pts = pts.reshape((-1, 1, 2))
                
                # Create a copy for overlay
                overlay = frame.copy()
                cv2.fillPoly(overlay, [pts], (0, 255, 0, 128))  # Green with transparency
                cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
                
                # Draw lane boundary
                cv2.polylines(frame, [pts], True, (0, 255, 0), 2)
                
                # Draw direction arrow
                cv2.arrowedLine(frame, arrow_start, arrow_end, (0, 255, 0), 3, tipLength=0.3)
                
                # Draw direction text
                text_pos = ((points[0][0] + points[-1][0]) // 2, (points[0][1] + points[-1][1]) // 2)
                cv2.putText(frame, direction_text, text_pos,
                          cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        for detection in detections:
            if detection.class_name in ['car', 'motorcycle', 'bus', 'truck'] and detection.track_id:
                # Check for wrong side driving
                if self._check_wrong_side_driving(detection, tracker_data, lane_zones):
                    # Create violation
                    violation = ViolationEvent(
                        violation_type="wrong_side_driving",
                        timestamp=datetime.now(),
                        track_id=detection.track_id,
                        location=(detection.bbox[0], detection.bbox[1]),
                        confidence=0.85,
                        evidence={
                            "bbox": detection.bbox,
                            "vehicle_type": detection.class_name,
                            "trajectory": tracker_data.get(detection.track_id, {}).get('positions', [])[-5:]
                        },
                        description=f"{detection.class_name} driving on wrong side"
                    )
                    violations.append(violation)
                    self.violation_count += 1
                    logger.warning(f"Wrong side driving detected: Track {detection.track_id}")
                    
                    # Draw wrong direction indicator
                    x1, y1, x2, y2 = detection.bbox
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(frame, "WRONG DIRECTION", (x1, y1 - 10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    
                    # Draw movement vector
                    if detection.track_id in tracker_data:
                        positions = tracker_data[detection.track_id]['positions']
                        if len(positions) >= 5:
                            start_pos = positions[-5]
                            end_pos = positions[-1]
                            cv2.arrowedLine(frame, start_pos, end_pos, (0, 0, 255), 2)
        
        return violations
    
    def _check_wrong_side_driving(self, detection: DetectionResult, tracker_data: Dict, 
                                lane_zones: Dict) -> bool:
        """Check if vehicle is driving on wrong side"""
        if not detection.track_id or detection.track_id not in tracker_data:
            return False
        
        # Need at least 5 positions for reliable direction detection
        track_positions = tracker_data[detection.track_id]['positions']
        if len(track_positions) < 5:
            return False
        
        # Get vehicle center
        x1, y1, x2, y2 = detection.bbox
        vehicle_center = ((x1 + x2) // 2, (y1 + y2) // 2)
        
        # Check if vehicle is in a lane zone
        for zone_id, zone_data in lane_zones.items():
            if zone_data['polygon'].contains(Point(vehicle_center[0], vehicle_center[1])):
                # Vehicle is in this lane zone
                # Get the lane direction from the zone ID
                # Format should be "lane_direction" where direction is "left" or "right"
                if "_left" in zone_id.lower():
                    expected_direction = "left"  # Vehicle should move left
                elif "_right" in zone_id.lower():
                    expected_direction = "right"  # Vehicle should move right
                else:
                    # Default to right if not specified
                    expected_direction = "right"
                
                # Calculate vehicle movement direction
                recent_positions = track_positions[-5:]
                start_x = recent_positions[0][0]
                end_x = recent_positions[-1][0]
                dx = end_x - start_x
                
                # Determine actual direction
                if dx > 20:  # Moving right
                    actual_direction = "right"
                elif dx < -20:  # Moving left
                    actual_direction = "left"
                else:
                    # Not enough horizontal movement
                    return False
                
                # Check if actual direction matches expected direction
                if actual_direction != expected_direction:
                    return True
        
        return False

# Removed HelmetSeatbeltViolationDetector, MobilePhoneViolationDetector, and OverloadingDetector as requested

class HeavyVehicleProhibitionDetector(BaseViolationDetector):
    """Detects heavy vehicle entry in prohibited zones"""
    
    def __init__(self):
        super().__init__("Heavy Vehicle Prohibition")
        self.heavy_vehicle_types = ['bus', 'truck']
        
    def detect(self, detections: List[DetectionResult], frame: np.ndarray, 
               tracker_data: Dict, zones: ZoneManager) -> List[ViolationEvent]:
        violations = []
        
        # Get prohibited zones for heavy vehicles
        prohibited_zones = zones.get_zones_by_type("heavy_vehicle_prohibited")
        
        # Only detect violations if prohibited zones are defined
        if not prohibited_zones:
            return violations
        
        for detection in detections:
            if (detection.class_name in self.heavy_vehicle_types and 
                self._check_prohibited_entry(detection, prohibited_zones)):
                
                violation = ViolationEvent(
                    violation_type="heavy_vehicle_prohibition",
                    timestamp=datetime.now(),
                    track_id=detection.track_id,
                    location=(detection.bbox[0], detection.bbox[1]),
                    confidence=0.9,
                    evidence={
                        "bbox": detection.bbox,
                        "vehicle_type": detection.class_name
                    },
                    description=f"{detection.class_name} entered prohibited zone"
                )
                violations.append(violation)
                self.violation_count += 1
                logger.warning(f"Heavy vehicle prohibition violated: Track {detection.track_id}")
        
        return violations
    
    def _check_prohibited_entry(self, detection: DetectionResult, prohibited_zones: Dict) -> bool:
        """Check if heavy vehicle entered prohibited zone"""
        center_x = (detection.bbox[0] + detection.bbox[2]) // 2
        center_y = (detection.bbox[1] + detection.bbox[3]) // 2
        
        for zone_id, zone_data in prohibited_zones.items():
            if zone_data['polygon'].contains(Point(center_x, center_y)):
                return True
        
        return False

class LaneChangeViolationDetector(BaseViolationDetector):
    """Detects illegal lane change violations"""
    
    def __init__(self):
        super().__init__("Lane Change Violation")
        self.vehicle_lane_history = {}  # Track lane history for each vehicle
        self.processed_violations = set()  # Prevent duplicate violations
        self.min_lane_time = 3.0  # Minimum time in lane before change allowed (seconds)
        
    def detect(self, detections: List[DetectionResult], frame: np.ndarray, 
               tracker_data: Dict, zones: ZoneManager) -> List[ViolationEvent]:
        violations = []
        
        # Get lane zones and no-lane-change zones
        lane_zones = zones.get_zones_by_type("lane_zone")
        no_change_zones = zones.get_zones_by_type("no_lane_change")
        
        # Only detect violations if lane zones are defined
        if not lane_zones:
            return violations
        
        current_time = time.time()
        
        # Draw lane zones
        self._draw_lane_zones(frame, lane_zones, no_change_zones)
        
        for detection in detections:
            if detection.class_name in ['car', 'motorcycle', 'bus', 'truck'] and detection.track_id:
                if self._check_lane_change_violation(detection, tracker_data, lane_zones, no_change_zones, current_time):
                    # Skip if already processed
                    if detection.track_id in self.processed_violations:
                        continue
                    
                    # Get lane change details
                    lane_history = self.vehicle_lane_history.get(detection.track_id, {})
                    from_lane = lane_history.get('previous_lane', 'Unknown')
                    to_lane = lane_history.get('current_lane', 'Unknown')
                    
                    violation = ViolationEvent(
                        violation_type="lane_change_violation",
                        timestamp=datetime.now(),
                        track_id=detection.track_id,
                        location=(detection.bbox[0], detection.bbox[1]),
                        confidence=0.85,
                        evidence={
                            "bbox": detection.bbox,
                            "vehicle_type": detection.class_name,
                            "from_lane": from_lane,
                            "to_lane": to_lane,
                            "trajectory": tracker_data.get(detection.track_id, {}).get('positions', [])[-10:]
                        },
                        description=f"{detection.class_name} illegal lane change from {from_lane} to {to_lane}"
                    )
                    violations.append(violation)
                    self.violation_count += 1
                    
                    # Mark as processed
                    self.processed_violations.add(detection.track_id)
                    
                    logger.warning(f"Lane change violation detected: Track {detection.track_id} - {from_lane} to {to_lane}")
                    
                    # Draw violation indicator
                    x1, y1, x2, y2 = detection.bbox
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                    cv2.putText(frame, "ILLEGAL LANE CHANGE", (x1, y1 - 10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        return violations
    
    def _draw_lane_zones(self, frame: np.ndarray, lane_zones: Dict, no_change_zones: Dict):
        """Draw lane zones and no-lane-change zones"""
        # Draw lane zones
        for zone_id, zone_data in lane_zones.items():
            points = zone_data['points']
            pts = np.array(points, np.int32)
            pts = pts.reshape((-1, 1, 2))
            
            # Different colors for different lanes
            lane_colors = {
                'lane_zone_1': (255, 100, 100),  # Light red
                'lane_zone_2': (100, 255, 100),  # Light green
                'lane_zone_3': (100, 100, 255),  # Light blue
                'lane_zone_4': (255, 255, 100),  # Light yellow
            }
            color = lane_colors.get(zone_id, (150, 150, 150))
            
            # Draw semi-transparent lane fill
            overlay = frame.copy()
            cv2.fillPoly(overlay, [pts], color)
            cv2.addWeighted(overlay, 0.2, frame, 0.8, 0, frame)
            
            # Draw lane boundary
            cv2.polylines(frame, [pts], True, color, 2)
            
            # Add lane label
            if points:
                lane_num = zone_id.split('_')[-1] if '_' in zone_id else '?'
                cv2.putText(frame, f"LANE {lane_num}", (points[0][0], points[0][1] - 10),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Draw no-lane-change zones
        for zone_id, zone_data in no_change_zones.items():
            points = zone_data['points']
            pts = np.array(points, np.int32)
            pts = pts.reshape((-1, 1, 2))
            
            # Red striped pattern for no-change zones
            cv2.polylines(frame, [pts], True, (0, 0, 255), 3)
            cv2.putText(frame, "NO LANE CHANGE", (points[0][0], points[0][1] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    
    def _check_lane_change_violation(self, detection: DetectionResult, tracker_data: Dict, 
                                   lane_zones: Dict, no_change_zones: Dict, current_time: float) -> bool:
        """Check if vehicle made illegal lane change"""
        if not detection.track_id or detection.track_id not in tracker_data:
            return False
        
        # Get vehicle center
        x1, y1, x2, y2 = detection.bbox
        vehicle_center = ((x1 + x2) // 2, (y1 + y2) // 2)
        vehicle_point = Point(vehicle_center[0], vehicle_center[1])
        
        # Determine current lane
        current_lane = None
        for zone_id, zone_data in lane_zones.items():
            if zone_data['polygon'].contains(vehicle_point):
                current_lane = zone_id
                break
        
        if not current_lane:
            return False  # Vehicle not in any lane
        
        # Initialize or update lane history
        if detection.track_id not in self.vehicle_lane_history:
            self.vehicle_lane_history[detection.track_id] = {
                'current_lane': current_lane,
                'previous_lane': None,
                'lane_entry_time': current_time,
                'lane_changes': 0
            }
            return False
        
        lane_history = self.vehicle_lane_history[detection.track_id]
        
        # Check if lane changed
        if current_lane != lane_history['current_lane']:
            # Lane change detected
            previous_lane = lane_history['current_lane']
            time_in_previous_lane = current_time - lane_history['lane_entry_time']
            
            # Check violation conditions
            violation_detected = False
            
            # 1. Check if in no-lane-change zone
            for zone_id, zone_data in no_change_zones.items():
                if zone_data['polygon'].contains(vehicle_point):
                    violation_detected = True
                    break
            
            # 2. Check minimum time in lane (prevent rapid lane switching)
            if time_in_previous_lane < self.min_lane_time:
                violation_detected = True
            
            # 3. Check excessive lane changes (zigzag driving)
            if lane_history['lane_changes'] >= 3:  # More than 3 changes in tracking period
                violation_detected = True
            
            # Update lane history
            lane_history['previous_lane'] = previous_lane
            lane_history['current_lane'] = current_lane
            lane_history['lane_entry_time'] = current_time
            lane_history['lane_changes'] += 1
            
            return violation_detected
        
        return False

class ParkingViolationDetector(BaseViolationDetector):
    """Detects parking violations"""
    
    def __init__(self):
        super().__init__("Parking Violation")
        self.stationary_threshold = 10  # Seconds (reduced from 30 to 10)
        self.stationary_vehicles = {}  # Track by ID
        self.stationary_positions = {}  # Track by position (more stable than IDs)
        
    def detect(self, detections: List[DetectionResult], frame: np.ndarray, 
               tracker_data: Dict, zones: ZoneManager) -> List[ViolationEvent]:
        violations = []
        
        # Get no-parking zones
        no_parking_zones = zones.get_zones_by_type("no_parking")
        
        # Only detect violations if no-parking zones are defined
        if not no_parking_zones:
            return violations
            
        current_time = time.time()
        
        # Draw no-parking zones with more visibility
        for zone_id, zone_data in no_parking_zones.items():
            points = zone_data['points']
            pts = np.array(points, np.int32)
            pts = pts.reshape((-1, 1, 2))
            
            # Create a copy of the frame for the overlay
            overlay = frame.copy()
            
            # Draw a semi-transparent red fill
            cv2.fillPoly(overlay, [pts], (0, 0, 255))  # Red fill
            
            # Blend the overlay with the original frame
            alpha = 0.2  # Transparency factor
            cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
            
            # Draw the outline and label (not transparent)
            cv2.polylines(frame, [pts], True, (0, 0, 255), 2)  # Red outline
            cv2.putText(frame, "NO PARKING ZONE", (points[0][0], points[0][1] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        # Draw position-based timers for vehicles being tracked for parking violations
        for position_key, data in self.stationary_positions.items():
            center_x, center_y = data['location']
            current_id = data['current_id']
            
            # Calculate how long the vehicle has been stationary
            stationary_duration = current_time - data['start_time']
            
            # Draw timer at the position
            cv2.putText(frame, f"Parked: {stationary_duration:.1f}s", 
                      (center_x - 50, center_y - 30),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)  # Orange
            
            # Draw circle at the position
            cv2.circle(frame, (center_x, center_y), 20, (0, 165, 255), 2)
            
            # Find the detection with the current ID
            for detection in detections:
                if detection.track_id == current_id:
                    x1, y1, x2, y2 = detection.bbox
                    
                    # Draw warning indicator if approaching violation threshold
                    if stationary_duration > self.stationary_threshold * 0.7:  # 70% of threshold
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 165, 255), 2)  # Orange
                    break
        
        for detection in detections:
            if detection.class_name in ['car', 'motorcycle', 'bus', 'truck'] and detection.track_id:
                if self._check_parking_violation(detection, tracker_data, no_parking_zones, current_time):
                    # Get position key for this detection
                    x1, y1, x2, y2 = detection.bbox
                    center_x = (x1 + x2) // 2
                    center_y = (y1 + y2) // 2
                    width = x2 - x1
                    height = y2 - y1
                    grid_size = 20
                    position_key = f"{center_x // grid_size}_{center_y // grid_size}_{width // 20}_{height // 20}"
                    
                    # Get the actual start time from position tracking
                    start_time = current_time
                    if position_key in self.stationary_positions:
                        start_time = self.stationary_positions[position_key]['start_time']
                    
                    stationary_duration = current_time - start_time
                    
                    violation = ViolationEvent(
                        violation_type="parking_violation",
                        timestamp=datetime.now(),
                        track_id=detection.track_id,
                        location=(detection.bbox[0], detection.bbox[1]),
                        confidence=0.85,
                        evidence={
                            "bbox": detection.bbox,
                            "vehicle_type": detection.class_name,
                            "stationary_duration": stationary_duration,
                            "position_key": position_key
                        },
                        description=f"{detection.class_name} parked in no-parking zone for {stationary_duration:.1f} seconds"
                    )
                    violations.append(violation)
                    self.violation_count += 1
                    logger.warning(f"Parking violation detected: Position {position_key}, Track {detection.track_id}")
        
        return violations
    
    def _check_parking_violation(self, detection: DetectionResult, tracker_data: Dict, 
                               no_parking_zones: Dict, current_time: float) -> bool:
        """Check for parking violation"""
        if not detection.track_id:
            return False
        
        # Get vehicle center and dimensions
        x1, y1, x2, y2 = detection.bbox
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        width = x2 - x1
        height = y2 - y1
        vehicle_point = Point(center_x, center_y)
        
        # Create a position key for this vehicle (grid cell)
        # This helps track vehicles even if their ID changes
        grid_size = 20  # pixels (reduced for better accuracy)
        position_key = f"{center_x // grid_size}_{center_y // grid_size}_{width // 20}_{height // 20}"
        
        # Check if vehicle is in no-parking zone
        in_no_parking_zone = False
        for zone_id, zone_data in no_parking_zones.items():
            if zone_data['polygon'].contains(vehicle_point):
                in_no_parking_zone = True
                break
        
        if not in_no_parking_zone:
            # Remove from stationary vehicles if not in no-parking zone
            if detection.track_id in self.stationary_vehicles:
                del self.stationary_vehicles[detection.track_id]
            
            # Also check position-based tracking
            if position_key in self.stationary_positions:
                del self.stationary_positions[position_key]
                
            return False
        
        # Check if vehicle is stationary based on tracking data
        is_stationary = True
        if detection.track_id in tracker_data:
            track_positions = tracker_data[detection.track_id]['positions']
            if len(track_positions) >= 5:  # Need at least 5 positions
                # Calculate movement in recent positions
                recent_positions = track_positions[-5:]  # Use last 5 positions
                max_movement = 0
                for i in range(len(recent_positions) - 1):
                    movement = ((recent_positions[i+1][0] - recent_positions[i][0])**2 + 
                              (recent_positions[i+1][1] - recent_positions[i][1])**2)**0.5
                    max_movement = max(max_movement, movement)
                
                # If significant movement detected, not stationary
                if max_movement >= 8:  # 8 pixels threshold
                    is_stationary = False
        
        # Update position-based tracking
        if is_stationary:
            # Start tracking by position if not already tracked
            if position_key not in self.stationary_positions:
                self.stationary_positions[position_key] = {
                    'start_time': current_time,
                    'location': (center_x, center_y),
                    'current_id': detection.track_id
                }
                logger.info(f"Vehicle at position {position_key} started parking in no-parking zone")
            else:
                # Update the current ID for this position
                self.stationary_positions[position_key]['current_id'] = detection.track_id
                
                # Check if stationary for too long
                stationary_duration = current_time - self.stationary_positions[position_key]['start_time']
                logger.info(f"Vehicle at position {position_key} (ID: {detection.track_id}) parked for {stationary_duration:.1f} seconds")
                
                # Also update ID-based tracking for visualization
                self.stationary_vehicles[detection.track_id] = {
                    'start_time': self.stationary_positions[position_key]['start_time'],
                    'location': (center_x, center_y),
                    'position_key': position_key
                }
                
                if stationary_duration > self.stationary_threshold:
                    # Continue generating violations every few seconds for persistent parking
                    # Only generate violation if enough time has passed since last violation
                    last_violation_time = self.stationary_positions[position_key].get('last_violation_time', 0)
                    if current_time - last_violation_time > 5.0:  # Generate violation every 5 seconds
                        self.stationary_positions[position_key]['last_violation_time'] = current_time
                        return True
        else:
            # Vehicle is moving, remove from stationary lists
            if detection.track_id in self.stationary_vehicles:
                position_key = self.stationary_vehicles[detection.track_id].get('position_key')
                if position_key and position_key in self.stationary_positions:
                    del self.stationary_positions[position_key]
                del self.stationary_vehicles[detection.track_id]
                logger.info(f"Vehicle {detection.track_id} moved, no longer considered parked")
        
        return False

# Violation detector registry
VIOLATION_DETECTORS = {
    'red_light': RedLightViolationDetector(),
    'wrong_side': WrongSideDrivingDetector(),
    'heavy_vehicle': HeavyVehicleProhibitionDetector(),
    'parking': ParkingViolationDetector(),
    'lane_change': LaneChangeViolationDetector()
}

def get_all_violations(detections: List[DetectionResult], frame: np.ndarray, 
                      tracker_data: Dict, zones: ZoneManager) -> List[ViolationEvent]:
    """Get all violations from all enabled detectors"""
    all_violations = []
    
    # Debug: Log detection info
    vehicle_detections = [d for d in detections if d.class_name in ['car', 'motorcycle', 'bus', 'truck']]
    
    for detector_name, detector in VIOLATION_DETECTORS.items():
        if detector.enabled:
            try:
                violations = detector.detect(detections, frame, tracker_data, zones)
                if violations:
                    logger.info(f"{detector_name} detector found {len(violations)} violations")
                all_violations.extend(violations)
            except Exception as e:
                logger.error(f"Error in {detector_name} detector: {e}")
    
    return all_violations

