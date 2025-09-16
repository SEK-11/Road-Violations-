import cv2
import argparse
from lane_aware_wrong_side import LaneAwareWrongSideDetector

try:
    from vehicle_tracker import VehicleTracker
except Exception as e:
    print(f"Warning: Could not load YOLOv8 tracker ({e}). Using simple tracker.")
    from simple_tracker import SimpleTracker as VehicleTracker

def main():
    parser = argparse.ArgumentParser(description='Lane-Aware Wrong-Side Detection')
    parser.add_argument('--input', type=str, required=True, help='Input video file path or camera index (0 for webcam)')
    parser.add_argument('--output', type=str, default='lane_wrong_side_violations', help='Output directory for violation images')
    args = parser.parse_args()
    
    # Open video source
    if args.input.isdigit():
        cap = cv2.VideoCapture(int(args.input))
    else:
        cap = cv2.VideoCapture(args.input)
    
    if not cap.isOpened():
        print(f"Error: Cannot open video source {args.input}")
        return
    
    # Read first frame for setup
    ret, first_frame = cap.read()
    if not ret:
        print("Error: Cannot read first frame")
        return
    
    # Initialize components
    vehicle_tracker = VehicleTracker()
    wrong_side_detector = LaneAwareWrongSideDetector(args.output)
    
    # Interactive setup
    print("Setting up lanes and directions...")
    wrong_side_detector.setup_lanes_and_directions(first_frame)
    
    # Reset video to beginning
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    frame_count = 0
    print("Starting lane-aware wrong-side detection...")
    print("Press 'q' to quit")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_count += 1
        
        # Detect and track vehicles
        vehicles = vehicle_tracker.detect_and_track(frame)
        
        # Detect wrong-side violations
        violations = wrong_side_detector.detect_violations(vehicles, frame.copy(), frame_count)
        
        # Draw visualization
        display_frame = frame.copy()
        
        # Draw lane setup (boundaries and directions)
        display_frame = wrong_side_detector.draw_setup(display_frame)
        
        # Draw vehicles
        for vehicle in vehicles:
            x1, y1, x2, y2 = vehicle['bbox']
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(display_frame, f"ID:{vehicle['id']}", 
                       (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        
        # Highlight violations
        for violation in violations:
            x1, y1, x2, y2 = violation['bbox']
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
            cv2.putText(display_frame, f"WRONG WAY - L{violation['lane']+1}!", 
                       (x1, y1-30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Draw motion vector
            center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
            motion = violation['motion_vector']
            end_x = int(center_x + motion[0] * 50)
            end_y = int(center_y + motion[1] * 50)
            cv2.arrowedLine(display_frame, (center_x, center_y), (end_x, end_y), (0, 0, 255), 2)
        
        # Display info
        cv2.putText(display_frame, f"Frame: {frame_count}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(display_frame, f"Vehicles: {len(vehicles)}", 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(display_frame, f"Wrong-way: {len(violations)}", 
                   (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.imshow('Lane-Aware Wrong-Side Detection', display_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    print("Detection completed.")

if __name__ == "__main__":
    main()