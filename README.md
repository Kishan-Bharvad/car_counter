# Vehicle Counter using YOLOv8 and SORT

This project detects, tracks, and counts vehicles in a prerecorded traffic video. It uses [Ultralytics YOLOv8](https://docs.ultralytics.com/) for object detection, the SORT algorithm for assigning a persistent ID to each detected vehicle, and OpenCV for video processing and display.

## Features

- Detects cars, bicycles, motorbikes, buses, and trucks.
- Tracks each detected vehicle with a unique ID.
- Counts a tracked vehicle once when it enters the configured counting area.
- Draws detection boxes, tracking IDs, vehicle centers, and the counting line.
- Displays the current total count while the video is running.
- Includes YOLOv8 nano and small model weights.

## How it works

For every video frame, YOLOv8 detects objects and returns their bounding boxes, classes, and confidence scores. Only supported vehicle classes are passed to SORT. SORT matches detections between frames and assigns a tracking ID to each vehicle. The center point of every tracked vehicle is compared with the configured counting region. An ID is added to the total only once, preventing the same tracked vehicle from being counted repeatedly.

```text
Input video
    |
    v
YOLOv8 vehicle detection
    |
    v
SORT ID tracking
    |
    v
Counting-region check
    |
    v
Annotated video and total count
```

## Project structure

```text
car_counter/
├── car_count.py        # Main detection and counting program
├── sort.py             # SORT multi-object tracking implementation
├── requirements.txt    # Python dependencies
├── car count.ipynb     # Notebook version of the project
├── yolov8n.pt          # YOLOv8 nano model used by default
├── yolov8s.pt          # YOLOv8 small model
├── video/              # Local input videos (excluded from Git)
└── README.md            # Project documentation
```

## Requirements

- Python 3.10 is recommended.
- macOS, Linux, or Windows.
- A local traffic video in MP4 format.
- A graphical desktop session for the OpenCV window.

The program uses the CPU when a supported GPU is unavailable. Processing speed depends on the computer, model, and video resolution.

## Installation

Clone the repository:

```bash
git clone https://github.com/Kishan-Bharvad/car_counter.git
cd car_counter
```

Create and activate a virtual environment.

On macOS or Linux:

```bash
python3 -m venv .env
source .env/bin/activate
```

On Windows PowerShell:

```powershell
py -m venv .env
.\.env\Scripts\Activate.ps1
```

Install the dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Add an input video

The current program reads this path:

```text
video/4.mp4
```

Create the directory and place a video there:

```bash
mkdir -p video
```

The sample videos are excluded from Git because they exceed GitHub's normal per-file size limit. To use a different file, change this line in `car_count.py`:

```python
cap = cv2.VideoCapture("video/4.mp4")
```

## Run the project

From the repository directory, with the virtual environment active, run:

```bash
python car_count.py
```

The annotated video opens in a full-screen OpenCV window. Press `h` to stop the program and close the window.

## Configuration

The default detector is the smaller and faster YOLOv8 nano model:

```python
model = YOLO("yolov8n.pt")
```

To use the more accurate but slower small model, change it to:

```python
model = YOLO("yolov8s.pt")
```

The counting line is configured in `car_count.py`:

```python
limits = [198, 500, 700, 398]
```

These values represent the start and end coordinates for the resized `880 x 740` frame. Adjust them for a different camera angle or video.

SORT is initialized with these settings:

```python
tracker = Sort(max_age=100, min_hits=3, iou_threshold=0.3)
```

- `max_age`: frames to retain a track without a matching detection.
- `min_hits`: detections required before a track is established.
- `iou_threshold`: minimum overlap used to match a detection with a track.

## Troubleshooting

### Video does not open

Confirm that `video/4.mp4` exists and run the command from the repository directory.

### SciPy fails to load on macOS

This project pins SciPy `1.13.1` for compatibility. Reinstall the pinned version:

```bash
python -m pip install --no-cache-dir --force-reinstall "scipy==1.13.1"
```

### The count is inaccurate

Counting depends on the camera angle, detector quality, tracking settings, and counting-line position. Adjust `limits`, try `yolov8s.pt`, or tune the SORT settings for the input video.

### Close the virtual environment

```bash
deactivate
```

## Notes

- Tracking IDs exist only for the current program run.
- Each tracking ID is counted once.
- The count is not saved to a file or database.
- Virtual environments, generated output, caches, and input videos are excluded through `.gitignore`.

## Repository

GitHub: [Kishan-Bharvad/car_counter](https://github.com/Kishan-Bharvad/car_counter)
