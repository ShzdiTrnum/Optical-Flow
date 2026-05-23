import numpy as np
import cv2
import matplotlib.pyplot as plt
import csv

# Optical flow calculation function
def flow(img1, img2):
    prvsImg = img1.copy()
    nextImg = img2.copy()
    flow = cv2.calcOpticalFlowFarneback(prvsImg, nextImg, None, 0.5, 3, 15, 5, 300, 1.0, 0)
    vx = flow[:, :, 0]
    vy = flow[:, :, 1]
    return vx, vy

# Rotation value calculation function
def rotation(vx, vy, cen_r, cen_c):
    r = (vy[cen_r, cen_c + 1] - vy[cen_r, cen_c]) - (vx[cen_r + 1, cen_c] - vx[cen_r, cen_c])
    return r

# Video settings
video_title = '/content/drive/MyDrive/Sh_CenterEstimate/fasciculation_simulation_accel_decel_test_new4.mp4'
cap = cv2.VideoCapture(video_title)

# Video parameter acquisition
fps = int(cap.get(cv2.CAP_PROP_FPS))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# Fixed frame range
frame_start = 0
frame_end   = 40
print(f"Total frames: {total_frames}, FPS: {fps}, Processing frames: {frame_start}–{frame_end}")

# Rectangle sizes: 10, 30, 50 ... 190
rect_sizes = range(10, 200, 20)
rotation_results = {size: [] for size in rect_sizes}

# Read first frame
ret, prev_frame = cap.read()
if not ret:
    raise ValueError("Failed to read the first frame of the video.")
prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

# Calculate rotation values
current_frame = 1

while current_frame <= frame_end:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    vx, vy = flow(prev_gray, gray)

    for size in rect_sizes:
        rect_rot_sum = 0
        cen_r = vx.shape[0] // 2
        cen_c = vx.shape[1] // 2
        for i in range(cen_r - size // 2, cen_r + size // 2):
            for j in range(cen_c - size // 2, cen_c + size // 2):
                rect_rot_sum += rotation(vx, vy, i, j)
        rotation_results[size].append(rect_rot_sum / (size ** 2))

    prev_gray = gray
    current_frame += 1

cap.release()
print(f"Processed {current_frame - 1} frame pairs.")

# Convert to NumPy arrays
for size in rect_sizes:
    rotation_results[size] = np.array(rotation_results[size])

# Moving average function
def moving_average(data, window_size):
    return np.convolve(data, np.ones(window_size) / window_size, mode='valid')

# Normalize results
normalized_results = {}
for size, values in rotation_results.items():
    moving_avg = moving_average(values, window_size=3)
    perimeter = 8 * size
    normalized_values = moving_avg / perimeter
    normalized_results[size] = np.concatenate(
        ([values[0] / perimeter], normalized_values, [values[-1] / perimeter])
    )

# Frame labels
num_results  = len(next(iter(normalized_results.values())))
frame_labels = range(frame_start, frame_start + num_results)

# Plot
fig, ax = plt.subplots(figsize=(16, 8))
scaled_results = {}
for size, normalized_values in normalized_results.items():
    scaled_values = normalized_values * size
    scaled_results[size] = scaled_values
    ax.plot(frame_labels, scaled_values, marker='o', markersize=2, linewidth=2.2, label=f"Size {size}px")

ax.set_title("Rotation Values for All Rectangular Sizes")
ax.set_xlabel("Frame Number")
ax.set_ylabel("Rotation Value")
ax.axhline(0, color='black', linestyle='--', linewidth=1)

# Legend outside the plot on the right
ax.legend(fontsize=8, loc='upper left', bbox_to_anchor=(1.01, 1), borderaxespad=0)

plt.tight_layout()
plt.show()

# Save results to CSV
csv_filename = f'Normalized_Rotation_Values_Frames_{frame_start}_{frame_end}.csv'
with open(csv_filename, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    headers = ['Frame'] + [f'Size {size}px' for size in rect_sizes]
    writer.writerow(headers)
    for i, frame_num in enumerate(frame_labels):
        row = [frame_num] + [scaled_results[size][i] for size in rect_sizes]
        writer.writerow(row)

print(f"CSV saved: {csv_filename}")
