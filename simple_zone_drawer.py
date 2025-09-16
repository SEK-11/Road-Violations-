#!/usr/bin/env python3
"""
Simple Zone Drawing Tool for Traffic Violation Detection System
This tool allows users to draw zones on a paused video frame and save them to a config file.
"""

import cv2
import numpy as np
import json
import argparse
from pathlib import Path

# Global variables
drawing = False
points = []
current_zone_type = "stop_line"
zones = []
frame = None
zone_colors = {
    'stop_line': (0, 0, 255),      # Red
    'lane': (0, 255, 0),           # Green
    'no_parking': (255, 0, 0),     # Blue
    'heavy_vehicle_prohibited': (0, 255, 255),  # Yellow
    'lane_divider': (0, 255, 255), # Yellow for lane divider lines
}


def mouse_callback(event, x, y, flags, param):
    """Mouse callback function for drawing zones"""
    global drawing, points, frame

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        points = [(x, y)]
    
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        
        # For lane dividers, only need 2 points (start and end of line)
        if current_zone_type == "lane_divider":
            if len(points) >= 2:
                # Use only first and last points for a straight line
                line_points = [points[0], points[-1]]
                zone_id = f"lane_divider_{len([z for z in zones if z['type'] == current_zone_type]) + 1}"
                
                zones.append({
                    "id": zone_id,
                    "type": current_zone_type,
                    "points": line_points
                })
                print(f"Added lane divider line: {zone_id}")
                points = []
        elif len(points) > 2:  # Need at least 3 points to form a polygon
            # Add the zone
            if current_zone_type == "lane":
                # For lanes, ask for direction
                print("\nFor lane zones, specify the direction:")
                print("1. Left-going lane (vehicles should move left)")
                print("2. Right-going lane (vehicles should move right)")
                direction = input("Enter 1 for left, 2 for right (default is right): ").strip()
                
                if direction == "1":
                    zone_id = f"lane_left_{len([z for z in zones if z['type'] == current_zone_type]) + 1}"
                else:
                    zone_id = f"lane_right_{len([z for z in zones if z['type'] == current_zone_type]) + 1}"
                    
                print(f"Created {zone_id} - vehicles should move {'left' if 'left' in zone_id else 'right'}")

            else:
                zone_id = f"{current_zone_type}_{len([z for z in zones if z['type'] == current_zone_type]) + 1}"
            
            zones.append({
                "id": zone_id,
                "type": current_zone_type,
                "points": points.copy()
            })
            print(f"Added {current_zone_type} zone with {len(points)} points")
            points = []
    
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            points.append((x, y))

def draw_zones(img):
    """Draw all zones on the image"""
    result = img.copy()
    
    # Draw current points being drawn
    if len(points) > 1:
        color = zone_colors.get(current_zone_type, (255, 255, 255))
        
        if current_zone_type == "lane_divider":
            # For lane dividers, draw a straight line from first to last point
            cv2.line(result, points[0], points[-1], color, 3)
        else:
            # For other zones, draw as polygon
            pts = np.array(points, np.int32)
            pts = pts.reshape((-1, 1, 2))
            cv2.polylines(result, [pts], False, color, 2)
    
    # Draw existing zones
    for zone in zones:
        color = zone_colors.get(zone['type'], (255, 255, 255))
        
        if zone['type'] == 'lane_divider':
            # Draw lane divider as a line
            if len(zone['points']) >= 2:
                cv2.line(result, zone['points'][0], zone['points'][-1], color, 3)
        else:
            # Draw other zones as polygons
            pts = np.array(zone['points'], np.int32)
            pts = pts.reshape((-1, 1, 2))
            cv2.polylines(result, [pts], True, color, 2)
        
        # Add zone label
        if zone['points']:
            label_pos = (zone['points'][0][0], zone['points'][0][1] - 10)
            cv2.putText(result, zone['id'], label_pos, 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    
    # Draw instructions
    instructions = [
        "VIDEO IS PAUSED FOR ZONE DRAWING",
        f"Current zone type: {current_zone_type}",
        "1:stop_line 2:lane 3:no_parking 4:heavy_vehicle 5:lane_divider",
        "Click and drag to draw. Press 5 for lane dividers (straight lines)",
        "For lanes, you'll be asked to specify direction (left or right)",
        "Press 's' to save and continue, 'c' to clear points, 'd' to delete last zone"
    ]
    
    # Add background for instructions
    cv2.rectangle(result, (0, 0), (600, 140), (0, 0, 0), -1)
    
    for i, text in enumerate(instructions):
        cv2.putText(result, text, (10, 20 + i * 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    return result

def save_config(output_file):
    """Save zones to config file"""
    # Store frame dimensions with the zones to ensure consistency
    if frame is not None:
        height, width = frame.shape[:2]
        frame_info = {"width": width, "height": height}
    else:
        frame_info = {"width": 1280, "height": 720}  # Default if no frame
    
    config = {
        "zones": zones,
        "frame_info": frame_info,
        "detectors": {
            "red_light": {"enabled": True},
            "heavy_vehicle": {"enabled": True},
            "parking": {"enabled": True},
            "lane_change": {"enabled": True}
        }
    }
    
    with open(output_file, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"Configuration saved to {output_file}")
    print(f"Frame dimensions: {frame_info['width']}x{frame_info['height']}")
    print("These dimensions will be used to scale zones in the main application.")
    print("Make sure to use the same video source in both tools for best results.")

def preprocess_frame(frame):
    """Apply the same preprocessing as in the main application"""
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

def main():
    parser = argparse.ArgumentParser(description='Simple Zone Drawing Tool')
    parser.add_argument('--source', type=str, default='0', 
                       help='Video source (0 for webcam, path for video file)')
    parser.add_argument('--output', type=str, default='config.json',
                       help='Output config file path')
    
    args = parser.parse_args()
    
    global zones, frame, current_zone_type
    
    # Initialize video capture
    source = args.source
    if source.isdigit():
        source = int(source)
    
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"Error: Could not open video source {source}")
        return
    
    # Create window and set mouse callback
    cv2.namedWindow('Zone Drawing Tool')
    cv2.setMouseCallback('Zone Drawing Tool', mouse_callback)
    
    # Read first frame
    ret, original_frame = cap.read()
    if not ret:
        print("Could not read frame from video source")
        return
        
    # Apply the same preprocessing as in the main application
    frame = preprocess_frame(original_frame)
    
    print("\n*** SIMPLE ZONE DRAWING TOOL ***")
    print("The video is paused for you to draw zones.")
    print("1. Draw stop lines (for red light violations)")
    print("2. Draw lanes (for wrong side driving)")
    print("3. Draw no-parking zones")
    print("4. Draw heavy vehicle prohibited zones")
    print("5. Draw lane divider lines (for line-based lane change detection)")


    print("When finished, press 's' to save and continue to the main application.\n")
    
    # Main loop - stay on the first frame for drawing
    while True:
        # Draw zones on frame
        display_frame = draw_zones(frame)
        
        # Show frame
        cv2.imshow('Zone Drawing Tool', display_frame)
        
        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            break
        elif key == ord('s'):
            save_config(args.output)
            print("Zones saved! You can now run the main application with:")
            print(f"python main_app.py --config {args.output}")
            break
        elif key == ord('c'):
            points.clear()
            print("Cleared current points")
        elif key == ord('d'):
            if zones:
                removed = zones.pop()
                print(f"Removed zone: {removed['id']}")
        elif key == ord('1'):
            current_zone_type = "stop_line"
            print(f"Current zone type: {current_zone_type}")
        elif key == ord('2'):
            current_zone_type = "lane"
            print(f"Current zone type: {current_zone_type}")
        elif key == ord('3'):
            current_zone_type = "no_parking"
            print(f"Current zone type: {current_zone_type}")
        elif key == ord('4'):
            current_zone_type = "heavy_vehicle_prohibited"
            print(f"Current zone type: {current_zone_type}")
        elif key == ord('5'):
            current_zone_type = "lane_divider"
            print(f"Current zone type: {current_zone_type} - Draw lines between lanes")


    
    # Clean up
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()