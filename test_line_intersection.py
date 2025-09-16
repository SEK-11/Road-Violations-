#!/usr/bin/env python3
"""
Test script for line intersection detection
"""

def line_intersection(p1, p2, p3, p4):
    """Check if line segment p1-p2 intersects with line segment p3-p4"""
    def ccw(A, B, C):
        return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])
    
    # Check if two line segments intersect
    return ccw(p1, p3, p4) != ccw(p2, p3, p4) and ccw(p1, p2, p3) != ccw(p1, p2, p4)

# Test cases
print("Testing line intersection function:")

# Test 1: Lines that cross
p1 = (0, 0)
p2 = (10, 10)
p3 = (0, 10)
p4 = (10, 0)
result = line_intersection(p1, p2, p3, p4)
print(f"Test 1 - Crossing lines: {result} (should be True)")

# Test 2: Lines that don't cross
p1 = (0, 0)
p2 = (5, 5)
p3 = (10, 0)
p4 = (15, 5)
result = line_intersection(p1, p2, p3, p4)
print(f"Test 2 - Non-crossing lines: {result} (should be False)")

# Test 3: Vehicle crossing lane divider (vertical line)
vehicle_prev = (100, 200)
vehicle_curr = (150, 200)
lane_start = (125, 100)
lane_end = (125, 300)
result = line_intersection(vehicle_prev, vehicle_curr, lane_start, lane_end)
print(f"Test 3 - Vehicle crossing vertical lane: {result} (should be True)")

# Test 4: Vehicle moving parallel to lane divider
vehicle_prev = (100, 200)
vehicle_curr = (150, 200)
lane_start = (100, 150)
lane_end = (150, 150)
result = line_intersection(vehicle_prev, vehicle_curr, lane_start, lane_end)
print(f"Test 4 - Vehicle parallel to lane: {result} (should be False)")

print("Line intersection tests completed!")