import numpy as np
import cv2
import matplotlib.pyplot as plt
import csv

def compute_optical_flow(previous_frame, current_frame):
    """
    Calculate dense optical flow between two consecutive frames
    Returns horizontal and vertical flow components
    """
    flow = cv2.calcOpticalFlowFarneback(
        previous_frame, current_frame, None, 0.5, 3, 15, 20, 5, 1.0, 0
    )
    return flow[:, :, 0], flow[:, :, 1]

def calculate_local_rotation(flow_x, flow_y, center_row, center_col):
    """
    Compute rotation component from optical flow derivatives
    Uses finite difference approximation of curl operation
    """
    dv_dx = flow_y[center_row, center_col + 1] - flow_y[center_row, center_col]
    du_dy = flow_x[center_row + 1, center_col] - flow_x[center_row, center_col]
    return dv_dx - du_dy

def apply_moving_average(signal, window_length=3):
    """Smooth signal using moving average filter"""
    return np.convolve(signal, np.ones(window_length) / window_length, mode='valid')

# Configuration parameters
VIDEO_PATH = '/content/drive/MyDrive/ALS - Trim20s.mp4'
REGION_OF_INTEREST = {
    'x_min': 250, 'x_max': 500,
    'y_min': 200, 'y_max': 450
}
ANALYSIS_RANGE = {
    'start_frame': 6,
    'end_frame': 24
}
RECTANGLE_DIMENSIONS = list(range(5, 101, 5))

# Initialize video capture and validate
video_capture = cv2.VideoCapture(VIDEO_PATH)
if not video_capture.isOpened():
    raise IOError(f"Cannot open video file: {VIDEO_PATH}")

# Storage for rotation analysis results
rotation_measurements = {size: [] for size in RECTANGLE_DIMENSIONS}

# Process first frame for initialization
success, initial_frame = video_capture.read()
if not success:
    raise ValueError("Unable to read initial video frame")

previous_grayscale = cv2.cvtColor(initial_frame, cv2.COLOR_BGR2GRAY)
frame_counter = 1

# Main processing loop
while True:
    success, current_frame = video_capture.read()
    if not success or frame_counter > ANALYSIS_RANGE['end_frame']:
        break

    if frame_counter >= ANALYSIS_RANGE['start_frame']:
        # Extract region of interest
        roi_previous = previous_grayscale[
            REGION_OF_INTEREST['y_min']:REGION_OF_INTEREST['y_max'],
            REGION_OF_INTEREST['x_min']:REGION_OF_INTEREST['x_max']
        ]
        roi_current = cv2.cvtColor(current_frame[
            REGION_OF_INTEREST['y_min']:REGION_OF_INTEREST['y_max'],
            REGION_OF_INTEREST['x_min']:REGION_OF_INTEREST['x_max']
        ], cv2.COLOR_BGR2GRAY)

        # Compute optical flow within ROI
        flow_x, flow_y = compute_optical_flow(roi_previous, roi_current)

        # Analyze rotation for each rectangle size
        for rect_size in RECTANGLE_DIMENSIONS:
            rotation_sum = 0
            center_row = flow_x.shape[0] // 2
            center_col = flow_x.shape[1] // 2

            # Sample rotation across the rectangle area
            for row in range(center_row - rect_size // 2, center_row + rect_size // 2):
                for col in range(center_col - rect_size // 2, center_col + rect_size // 2):
                    rotation_sum += calculate_local_rotation(flow_x, flow_y, row, col)

            rotation_measurements[rect_size].append(rotation_sum / (rect_size ** 2))

        previous_grayscale = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)

    frame_counter += 1

# Clean up video resources
video_capture.release()
cv2.destroyAllWindows()

# Convert results to numpy arrays
for size in RECTANGLE_DIMENSIONS:
    rotation_measurements[size] = np.array(rotation_measurements[size])

# Process and normalize rotation data
processed_results = {}
radius_scaled_results = {}

for size, raw_values in rotation_measurements.items():
    # Smooth data and normalize by perimeter
    smoothed_values = apply_moving_average(raw_values, 3)
    normalization_factor = 8 * size  # Perimeter approximation
    normalized_data = smoothed_values / normalization_factor

    # Reconstruct full length array
    processed_results[size] = np.concatenate((
        [raw_values[0] / normalization_factor],
        normalized_data,
        [raw_values[-1] / normalization_factor]
    ))

    # Scale by radius for final output
    radius_scaled_results[size] = processed_results[size] * size

# Visualize radius-scaled rotation values
plt.figure(figsize=(14, 8))

for size, scaled_data in radius_scaled_results.items():
    frame_numbers = range(ANALYSIS_RANGE['start_frame'], ANALYSIS_RANGE['end_frame'] + 1)
    plt.plot(frame_numbers, scaled_data, marker='o', label=f"r= {size}")

plt.xlabel("Frame Number", fontsize=20)
plt.ylabel("Rotation Value (Scaled by Radius)", fontsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.axhline(0, color='black', linestyle='--', linewidth=2)
plt.legend(
    fontsize=16,
    loc='center left',
    bbox_to_anchor=(1.02, 0.5),
    borderaxespad=0.
)
plt.tight_layout(rect=[0, 0, 0.8, 1])
plt.show()

# Export results to CSV
output_filename = f'Normalized Rotation Values (Scaled by Radius) Frames {ANALYSIS_RANGE["start_frame"]}-{ANALYSIS_RANGE["end_frame"]}.csv'

with open(output_filename, 'w', newline='') as output_file:
    csv_writer = csv.writer(output_file)

    # Write column headers
    header_row = ['Frame'] + [f'Size {size}px' for size in RECTANGLE_DIMENSIONS]
    csv_writer.writerow(header_row)

    # Write data rows
    for index, frame_num in enumerate(range(ANALYSIS_RANGE['start_frame'], ANALYSIS_RANGE['end_frame'] + 1)):
        data_row = [frame_num] + [radius_scaled_results[size][index] for size in RECTANGLE_DIMENSIONS]
        csv_writer.writerow(data_row)
