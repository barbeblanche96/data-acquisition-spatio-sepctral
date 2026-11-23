# Event-driven acquisition of raw spatio-spectral images

Acquisition script associated with the paper *"Tracking-based recognition framework using 2D raw spatio-spectral images for free-flying pest insect monitoring"*.

The script records raw 2D spatio-spectral images only while motion is detected in the scene. It was designed for a XIMEA xiQ MQ022HG-IM-LS150 VISNIR camera driven by an NVIDIA Jetson TX2.

## Important note

This repository contains a **re-implementation** of the acquisition strategy described in the paper. The script that was deployed in the field is no longer available. This version was rebuilt from an earlier acquisition script and from the description and parameter values published in the paper.

It has been checked offline on synthetic image sequences (`--replay` mode). It has **not** been run again on the camera and the embedded platform.

## Principle

Two parallel processes communicate through a shared FIFO queue:

1. The acquisition process grabs frames from the camera API and pushes them into the queue.
2. The detection process feeds each frame to a KNN background model, refines the binary mask and validates the moving objects.

A frame is recorded, together with its mask, when at least one validated object is present. Recording stops as soon as the scene becomes static again, and a new motion event starts a new sequence.

## Parameters

| Parameter | Value |
|---|---|
| Frames in the pixel history | 250 |
| Number of nearest neighbours | 17 |
| Distance threshold (8-bit intensity) | 11 |
| Structuring element (opening, then closing) | 15 × 15 |
| Accepted object area | 1% to 15% of the image area |

OpenCV expects a squared distance, so the threshold is passed as `dist2Threshold = 11² = 121`.

One implementation detail is not a parameter of the paper: the first 600 frames (about 30 s) are used only to initialise the background model, and nothing is recorded during this period (`--warmup`).

## Usage

```bash
pip install -r requirements.txt

# Acquisition from the camera for one hour
python acquisition_event_driven.py -d 3600 -e 40000 -g 0

# Acquisition until Ctrl-C
python acquisition_event_driven.py -d -1

# Offline check on a folder of recorded images, without the camera
python acquisition_event_driven.py --replay path/to/images
```

## Output

```
acquisitions/
└── <ddmmyy_HHMMSS>/            one folder per run
    ├── sequence_0001/          one folder per motion event
    │   ├── <timestamp>.bmp     raw image, without annotation
    │   ├── <timestamp>.bmp.npy binary mask of the validated objects (0 or 255)
    │   └── ...
    └── sequence_0002/
```

The mask is computed on the image without its blank areas (the 120 rows of the dead band and 4 rows at the top and at the bottom), so its size is 960 × 2048 for a 1088 × 2048 raw image.
