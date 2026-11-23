#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Event-driven acquisition of raw 2D spatio-spectral images based on motion detection.

Two parallel processes communicate through a shared FIFO queue:
  1. the acquisition process (main) grabs frames from the XIMEA camera API;
  2. the detection process feeds each frame to a KNN background model, refines
     the binary mask, validates the moving objects and records the frame and its
     mask only while motion is present.

Each motion event is stored in its own sequence folder. Recording stops as soon
as the scene becomes static again; a new motion event starts a new sequence.

NOTE: this file is a re-implementation of the acquisition strategy described in
the paper. The script deployed in the field is no longer available; this version
was rebuilt from an earlier acquisition script and from the published
description. See README.md.
"""

import argparse
import datetime
import multiprocessing
import os
import signal
import time
from time import localtime, strftime

import cv2
import numpy as np
import PIL.Image

# --------------------------------------------------------------------------- #
# Parameters reported in the paper
# --------------------------------------------------------------------------- #
HISTORY = 250            # number of frames stored in the pixel history
KNN_SAMPLES = 17         # number of close neighbours required for a background decision
DIST_THRESHOLD = 11      # distance threshold on pixel intensity (8-bit scale, max 255)
KERNEL_SIZE = 15         # size of the square structuring element (opening, then closing)
MIN_AREA_RATIO = 0.01    # an object must cover at least 1% of the image area
MAX_AREA_RATIO = 0.15    # ... and at most 15% of the image area

# --------------------------------------------------------------------------- #
# Sensor layout and camera handling
# --------------------------------------------------------------------------- #
DEAD_BAND_ROWS = np.s_[324:444]  # 120 blank rows between the VIS and NIR sensor regions
BORDER_ROWS = 4                  # unused rows at the top and at the bottom of the sensor
UNLOAD_EVERY = 1600              # the camera buffer is reset every UNLOAD_EVERY frames

# Implementation detail (not a parameter of the paper): number of initial frames used
# only to initialise the background model. Nothing is recorded during this period,
# because the model labels most pixels as foreground until its history is filled.
WARMUP_FRAMES = 600


def remove_blank_area(image):
    """Remove the dead band and the unused border rows of a raw image."""
    cropped = np.delete(image, DEAD_BAND_ROWS, axis=0)
    return cropped[BORDER_ROWS:-BORDER_ROWS, :]


def create_background_model():
    """KNN background model configured with the parameters of the paper."""
    # OpenCV expects a squared distance threshold.
    model = cv2.createBackgroundSubtractorKNN(
        history=HISTORY,
        dist2Threshold=float(DIST_THRESHOLD ** 2),
        detectShadows=False,
    )
    model.setkNNSamples(KNN_SAMPLES)
    return model


def detect_motion(image, model, kernel):
    """
    Label the pixels of a raw image and validate the moving objects.

    Returns the refined binary mask (uint8, 0 or 255), restricted to the
    validated objects, and the number of validated objects. The mask has the
    size of the image without its blank areas.
    """
    mask = model.apply(remove_blank_area(image))

    # Opening removes small noisy regions, closing homogenises the object regions.
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # Area-based filtering of the extracted contours.
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    image_area = mask.size
    valid = [
        contour for contour in contours
        if MIN_AREA_RATIO <= cv2.contourArea(contour) / image_area <= MAX_AREA_RATIO
    ]

    refined = np.zeros_like(mask)
    if valid:
        cv2.drawContours(refined, valid, -1, 255, thickness=cv2.FILLED)
    return refined, len(valid)


def record_images(image_queue, output_dir, warmup_frames=WARMUP_FRAMES):
    """Detection process: consume the FIFO queue and record the motion events."""
    # Ctrl-C is handled by the acquisition process, which then sends the stop item.
    signal.signal(signal.SIGINT, signal.SIG_IGN)

    model = create_background_model()
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (KERNEL_SIZE, KERNEL_SIZE))

    recording = False
    sequence_index = 0
    sequence_dir = None
    n_processed = 0

    while True:
        image, filename = image_queue.get()
        if image is None:  # stop item
            break

        mask, n_objects = detect_motion(image, model, kernel)
        n_processed += 1
        if n_processed <= warmup_frames:
            continue  # background model initialisation

        if n_objects > 0:
            if not recording:
                # A new motion event starts a new sequence.
                sequence_index += 1
                sequence_dir = os.path.join(output_dir, "sequence_{:04d}".format(sequence_index))
                os.makedirs(sequence_dir, exist_ok=True)
                recording = True
            # The raw image is saved without annotation, together with its mask.
            PIL.Image.fromarray(image).save(os.path.join(sequence_dir, filename))
            np.save(os.path.join(sequence_dir, filename), mask)  # -> <filename>.npy
        else:
            # The scene is static again: the current sequence is closed.
            recording = False

    print("{} sequence(s) recorded in {}".format(sequence_index, output_dir))


def configure_camera(cam, exposure, gain):
    cam.set_imgdataformat('XI_RAW8')
    cam.set_buffer_policy('XI_BP_SAFE')
    cam.set_gain(gain)
    cam.set_exposure(exposure)


def acquire_from_camera(image_queue, duration, exposure, gain):
    """Acquisition process: push every camera frame into the FIFO queue."""
    from ximea import xiapi  # imported here so that --replay works without the camera API

    stop_requested = {"value": False}

    def request_stop(signum, frame):
        print("Stopping the acquisition...")
        stop_requested["value"] = True

    signal.signal(signal.SIGINT, request_stop)

    cam = xiapi.Camera()
    print("Opening camera...")
    cam.open_device()
    configure_camera(cam, exposure, gain)
    cam.start_acquisition()

    img = xiapi.Image()
    begin = time.time()
    n_frames = 0
    try:
        while not stop_requested["value"] and (duration < 0 or (time.time() - begin) < duration):
            cam.get_image(img)
            filename = str(datetime.datetime.now().timestamp()) + ".bmp"
            image_queue.put((img.get_image_data_numpy(), filename))

            n_frames += 1
            if n_frames == UNLOAD_EVERY:
                # Reset the camera to release its buffer memory.
                cam.stop_acquisition()
                cam.close_device()
                cam.open_device()
                configure_camera(cam, exposure, gain)
                cam.start_acquisition()
                n_frames = 0
    finally:
        cam.stop_acquisition()
        cam.close_device()


def replay_from_folder(image_queue, folder):
    """Offline mode: push previously recorded images into the FIFO queue."""
    extensions = (".bmp", ".png", ".tif", ".tiff")
    filenames = sorted(f for f in os.listdir(folder) if f.lower().endswith(extensions))
    for name in filenames:
        image = np.array(PIL.Image.open(os.path.join(folder, name)))
        image_queue.put((image, os.path.splitext(name)[0] + ".bmp"))
    print("{} image(s) replayed from {}".format(len(filenames), folder))


def main():
    parser = argparse.ArgumentParser(description="Event-driven spatio-spectral acquisition")
    parser.add_argument("-d", "--duration", type=int, default=60,
                        help="acquisition duration in seconds (negative value: until Ctrl-C)")
    parser.add_argument("-e", "--exposure", type=int, default=40000,
                        help="camera exposure time in microseconds")
    parser.add_argument("-g", "--gain", type=float, default=0.0,
                        help="camera gain in dB")
    parser.add_argument("-o", "--output", type=str, default="acquisitions",
                        help="parent folder of the recorded sequences")
    parser.add_argument("--warmup", type=int, default=WARMUP_FRAMES,
                        help="number of initial frames used only to initialise the background model")
    parser.add_argument("--replay", type=str, default=None,
                        help="run the detection on a folder of recorded images instead of the camera")
    args = parser.parse_args()

    output_dir = os.path.join(args.output, strftime("%d%m%y_%H%M%S", localtime()))
    os.makedirs(output_dir, exist_ok=True)

    image_queue = multiprocessing.Queue()
    process = multiprocessing.Process(target=record_images, args=(image_queue, output_dir, args.warmup))
    process.start()

    try:
        if args.replay is not None:
            replay_from_folder(image_queue, args.replay)
        else:
            acquire_from_camera(image_queue, args.duration, args.exposure, args.gain)
    finally:
        image_queue.put((None, None))  # stop item
        process.join()


if __name__ == "__main__":
    main()
