import cv2
import numpy as np
from collections import defaultdict, deque

class LaneAwareWrongSideDetector:
    def __init__(self, output_dir="violations"):
        self.vehicle_trajectories = defaultdict(lambda: deque(maxlen=10))
        self.lane_flows = {}  # Store flow direction for each lane
        self.lane_boundaries = []
        self.output_dir = output_dir
        self.setup_complete = False
        
    def setup_lanes_and_directions(self, frame):
        """Interactive setup to define lanes and their correct directions"""
        self.lane_boundaries = []
        self.lane_flows = {}
        
        setup_frame = frame.copy()
        cv2.namedWindow('Setup Lanes and Directions')
        
        print("Lane Setup Instructions:")
        print("1. First, draw lane separation lines (click and drag)")
        print("2. Then define direction for each lane by drawing arrows")
        print("3. Press 's' to save and continue")
        
        # Step 1: Draw lane boundaries
        self._draw_lane_boundaries(setup_frame)
        
        # Step 2: Define directions for each lane
        if self.lane_boundaries:
            self._define_lane_directions(setup_frame)
        
        cv2.destroyWindow('Setup Lanes and Directions')
        self.setup_complete = True
        
    def _draw_lane_boundaries(self, frame):
        """Draw lane separation lines"""
        drawing = False
        current_line = []
        
        def mouse_callback(event, x, y, flags, param):
            nonlocal drawing, current_line
            if event == cv2.EVENT_LBUTTONDOWN:
                drawing = True
                current_line = [(x, y)]
            elif event == cv2.EVENT_LBUTTONUP and drawing:
                current_line.append((x, y))
                self.lane_boundaries.append(current_line.copy())
                drawing = False
                print(f"Lane boundary {len(self.lane_boundaries)} added")
        
        cv2.setMouseCallback('Setup Lanes and Directions', mouse_callback)
        
        while True:
            display_frame = frame.copy()
            
            # Draw existing boundaries
            for i, boundary in enumerate(self.lane_boundaries):
                if len(boundary) == 2:
                    cv2.line(display_frame, boundary[0], boundary[1], (0, 255, 0), 3)
                    cv2.putText(display_frame, f"B{i+1}", boundary[0], 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            cv2.putText(display_frame, "Draw lane boundaries, then press SPACE", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(display_frame, f"Boundaries: {len(self.lane_boundaries)}", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            cv2.imshow('Setup Lanes and Directions', display_frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord(' '):  # Space to continue
                break
            elif key == ord('c'):  # Clear
                self.lane_boundaries = []
    
    def _define_lane_directions(self, frame):
        """Define correct direction for each lane"""
        current_lane = 0
        direction_arrows = []
        
        def mouse_callback(event, x, y, flags, param):
            nonlocal current_lane, direction_arrows
            if event == cv2.EVENT_LBUTTONDOWN and current_lane < len(self._get_lane_regions()):
                direction_arrows.append([(x, y)])
            elif event == cv2.EVENT_LBUTTONUP and current_lane < len(self._get_lane_regions()):
                if len(direction_arrows) > current_lane and len(direction_arrows[current_lane]) == 1:
                    direction_arrows[current_lane].append((x, y))
                    
                    # Calculate direction vector
                    start, end = direction_arrows[current_lane]
                    dx = end[0] - start[0]
                    dy = end[1] - start[1]
                    magnitude = np.sqrt(dx**2 + dy**2)
                    
                    if magnitude > 0:
                        direction = (dx / magnitude, dy / magnitude)
                        self.lane_flows[current_lane] = direction
                        print(f"Lane {current_lane + 1} direction set")
                        current_lane += 1
        
        cv2.setMouseCallback('Setup Lanes and Directions', mouse_callback)
        
        while current_lane < len(self._get_lane_regions()):
            display_frame = frame.copy()
            
            # Draw lane boundaries
            for i, boundary in enumerate(self.lane_boundaries):
                if len(boundary) == 2:
                    cv2.line(display_frame, boundary[0], boundary[1], (0, 255, 0), 2)
            
            # Highlight current lane
            lane_regions = self._get_lane_regions()
            if current_lane < len(lane_regions):
                cv2.putText(display_frame, f"Draw arrow for LANE {current_lane + 1} direction", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            # Draw existing direction arrows
            for i, arrow in enumerate(direction_arrows):
                if len(arrow) == 2:
                    cv2.arrowedLine(display_frame, arrow[0], arrow[1], (255, 0, 0), 3)
                    cv2.putText(display_frame, f"L{i+1}", arrow[0], 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            
            cv2.imshow('Setup Lanes and Directions', display_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    def _get_lane_regions(self):
        """Get lane regions based on boundaries"""
        if not self.lane_boundaries:
            return [0]  # Single lane if no boundaries
        
        # Number of lanes = number of boundaries + 1
        return list(range(len(self.lane_boundaries) + 1))
    
    def _get_vehicle_lane(self, vehicle_center, frame_width):
        """Determine which lane a vehicle is in"""
        if not self.lane_boundaries:
            return 0
        
        # Get x-coordinates of boundaries at bottom of frame
        boundary_x_coords = []
        for boundary in self.lane_boundaries:
            if len(boundary) == 2:
                (x1, y1), (x2, y2) = boundary
                # Extend to bottom of frame
                if y2 != y1:
                    slope = (x2 - x1) / (y2 - y1)
                    x_bottom = int(x1 + slope * (frame_width - 1 - y1))
                else:
                    x_bottom = x1
                boundary_x_coords.append(x_bottom)
        
        boundary_x_coords.sort()
        
        # Determine lane based on vehicle center x-coordinate
        vehicle_x = vehicle_center[0]
        for i, boundary_x in enumerate(boundary_x_coords):
            if vehicle_x < boundary_x:
                return i
        return len(boundary_x_coords)  # Rightmost lane
    
    def detect_violations(self, vehicles, frame, frame_count):
        if not self.setup_complete:
            return []
        
        violations = []
        
        for vehicle in vehicles:
            vehicle_id = vehicle['id']
            center = vehicle['center']
            
            # Store trajectory
            self.vehicle_trajectories[vehicle_id].append(center)
            
            # Get vehicle's lane
            vehicle_lane = self._get_vehicle_lane(center, frame.shape[1])
            
            # Calculate motion vector
            if len(self.vehicle_trajectories[vehicle_id]) >= 3:
                motion_vector = self._calculate_motion_vector(vehicle_id)
                
                if motion_vector is not None and vehicle_lane in self.lane_flows:
                    # Check if motion is against lane direction
                    if self._is_wrong_direction(motion_vector, vehicle_lane):
                        violation_info = {
                            'vehicle_id': vehicle_id,
                            'bbox': vehicle['bbox'],
                            'lane': vehicle_lane,
                            'motion_vector': motion_vector,
                            'frame_count': frame_count
                        }
                        violations.append(violation_info)
                        self._save_violation(frame, violation_info)
        
        return violations
    
    def _calculate_motion_vector(self, vehicle_id):
        trajectory = list(self.vehicle_trajectories[vehicle_id])
        if len(trajectory) < 3:
            return None
        
        # Use last 3 points
        recent_points = trajectory[-3:]
        dx_total = recent_points[-1][0] - recent_points[0][0]
        dy_total = recent_points[-1][1] - recent_points[0][1]
        
        distance = np.sqrt(dx_total**2 + dy_total**2)
        if distance < 15:  # Minimum movement threshold
            return None
        
        magnitude = np.sqrt(dx_total**2 + dy_total**2)
        if magnitude > 0:
            return (dx_total / magnitude, dy_total / magnitude)
        return None
    
    def _is_wrong_direction(self, motion_vector, lane):
        """Check if vehicle motion is against lane direction"""
        if lane not in self.lane_flows:
            return False
        
        lane_direction = self.lane_flows[lane]
        
        # Calculate angle between motion and lane direction
        dot_product = (motion_vector[0] * lane_direction[0] + 
                      motion_vector[1] * lane_direction[1])
        dot_product = max(-1.0, min(1.0, dot_product))
        angle_diff = np.arccos(dot_product) * 180 / np.pi
        
        # If angle > 90 degrees, vehicle is moving wrong way
        return angle_diff > 90
    
    def _save_violation(self, frame, violation_info):
        from datetime import datetime
        import os
        
        os.makedirs(self.output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"wrong_side_lane{violation_info['lane']}_{violation_info['vehicle_id']}_{timestamp}.jpg"
        filepath = os.path.join(self.output_dir, filename)
        
        # Draw violation
        x1, y1, x2, y2 = violation_info['bbox']
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
        cv2.putText(frame, f"WRONG WAY - Lane {violation_info['lane']+1}", 
                   (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        cv2.imwrite(filepath, frame)
        print(f"Wrong-way violation saved: Lane {violation_info['lane']+1}, Vehicle {violation_info['vehicle_id']}")
    
    def draw_setup(self, frame):
        """Draw lane setup on frame"""
        # Draw boundaries
        for i, boundary in enumerate(self.lane_boundaries):
            if len(boundary) == 2:
                cv2.line(frame, boundary[0], boundary[1], (0, 255, 0), 2)
        
        # Draw direction arrows for each lane
        for lane_id, direction in self.lane_flows.items():
            # Calculate arrow position for each lane
            if lane_id < len(self._get_lane_regions()):
                # Position arrow in middle of lane
                frame_width = frame.shape[1]
                if self.lane_boundaries:
                    if lane_id == 0:
                        lane_center_x = frame_width // (len(self.lane_boundaries) + 1) * (lane_id + 1)
                    else:
                        lane_center_x = frame_width // (len(self.lane_boundaries) + 1) * (lane_id + 1)
                else:
                    lane_center_x = frame_width // 2
                
                arrow_start = (lane_center_x, 100)
                arrow_end = (int(arrow_start[0] + direction[0] * 80), 
                           int(arrow_start[1] + direction[1] * 80))
                
                cv2.arrowedLine(frame, arrow_start, arrow_end, (255, 0, 0), 3)
                cv2.putText(frame, f"L{lane_id+1}", 
                           (arrow_start[0]-15, arrow_start[1]-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        
        return frame