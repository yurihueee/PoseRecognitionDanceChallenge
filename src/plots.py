import argparse
import pathlib
import random

import cv2
import matplotlib.pyplot as plt
import numpy as np
from fastdtw import fastdtw
from scipy.spatial.distance import cosine

import extractor as ex


def add_noise_to_video(video_path, variance=0.1):
    """Add Gaussian noise to a video and save the output as a new video file"""
    cap = cv2.VideoCapture(video_path)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_path = str(pathlib.Path(video_path).with_suffix("")) + "_noisy.mov"
    out = cv2.VideoWriter(
        out_path,
        fourcc,
        cap.get(cv2.CAP_PROP_FPS),
        (
            int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        ),
    )

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # Apply Gaussian noise
        noise = np.random.normal(0, variance**0.5, frame.shape).astype("uint8")
        noisy_frame = cv2.add(frame, noise)
        out.write(noisy_frame)

    cap.release()
    out.release()
    return out_path


def scale_time(video_path, scale_factor=1.5):
    """Create a time-scaled version of the video"""
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS) * scale_factor)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_path = (
        str(pathlib.Path(video_path).with_suffix("")) + f"_scaled_{scale_factor}.mov"
    )
    out = cv2.VideoWriter(
        out_path,
        fourcc,
        fps,
        (
            int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        ),
    )

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame)

    cap.release()
    out.release()
    return out_path


def create_shifted_video(video_path, shift_seconds):
    """Shift the video by a certain number of seconds"""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    shift_frames = int(fps * shift_seconds)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_path = (
        str(pathlib.Path(video_path).with_suffix("")) + f"_shifted_{shift_seconds}.mov"
    )
    out = cv2.VideoWriter(
        out_path,
        fourcc,
        fps,
        (
            int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        ),
    )

    # Skip frames to create a delay
    for _ in range(shift_frames):
        cap.read()

    # Write remaining frames
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame)

    # Loop to the start and write the skipped frames
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    for _ in range(shift_frames):
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame)

    cap.release()
    out.release()
    return out_path


def analyze_video(original_video, manipulated_video):
    """Compare the original video with a manipulated version using FastDTW"""
    cap_original = cv2.VideoCapture(original_video)
    cap_manipulated = cv2.VideoCapture(manipulated_video)
    detector = ex.poseDetector()

    dtw_distances = []
    while True:
        ret_orig, frame_orig = cap_original.read()
        ret_manip, frame_manip = cap_manipulated.read()
        if not ret_orig or not ret_manip:
            break

        # Process frames
        frame_orig = detector.findPose(frame_orig, draw=False)
        lm_original = detector.findPosition(frame_orig, draw=False)
        frame_manip = detector.findPose(frame_manip, draw=False)
        lm_manipulated = detector.findPosition(frame_manip, draw=False)

        # Compute DTW distance
        if lm_original and lm_manipulated:
            distance, _ = fastdtw(lm_original, lm_manipulated, dist=cosine)
            dtw_distances.append(distance)

    cap_original.release()
    cap_manipulated.release()
    return dtw_distances


def plot_dtw_analysis(dtw_distances, title, output_path):
    """Plot DTW distances and save the plot to a file"""
    plt.figure()
    plt.plot(dtw_distances, label="DTW Distance")
    plt.title(title)
    plt.xlabel("Frame")
    plt.ylabel("Distance")
    plt.legend()
    plt.savefig(output_path)  # Save the figure
    plt.close()  # Close the figure after saving to free up memory


def main():
    parser = argparse.ArgumentParser(
        description="Analyze video manipulations with FastDTW."
    )
    parser.add_argument("video_path", help="Path to the original video file")
    args = parser.parse_args()

    # Create manipulated videos
    noisy_video = add_noise_to_video(args.video_path)
    scaled_video = scale_time(args.video_path, scale_factor=1.5)
    shifted_video = create_shifted_video(args.video_path, shift_seconds=2)

    # Analyze videos
    dtw_noisy = analyze_video(args.video_path, noisy_video)
    dtw_scaled = analyze_video(args.video_path, scaled_video)
    dtw_shifted = analyze_video(args.video_path, shifted_video)
    dtw_original = analyze_video(args.video_path, args.video_path)

    # Define plot file paths
    plot_path_noisy = (
        str(pathlib.Path(args.video_path).with_suffix("")) + "_DTW_Noisy.png"
    )
    plot_path_scaled = (
        str(pathlib.Path(args.video_path).with_suffix("")) + "_DTW_Scaled.png"
    )
    plot_path_shifted = (
        str(pathlib.Path(args.video_path).with_suffix("")) + "_DTW_Shifted.png"
    )
    plot_path_original = (
        str(pathlib.Path(args.video_path).with_suffix("")) + "_DTW_Original.png"
    )

    # Plot and save analysis
    plot_dtw_analysis(dtw_noisy, "DTW Distance for Noisy Video", plot_path_noisy)
    plot_dtw_analysis(
        dtw_scaled, "DTW Distance for Time-Scaled Video", plot_path_scaled
    )
    plot_dtw_analysis(
        dtw_shifted, "DTW Distance for Time-Shifted Video", plot_path_shifted
    )
    plot_dtw_analysis(
        dtw_original, "DTW Distance for Original Video", plot_path_original
    )


if __name__ == "__main__":
    main()
