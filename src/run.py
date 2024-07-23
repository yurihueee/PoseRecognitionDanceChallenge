import argparse
import logging
import time
from ffpyplayer.player import MediaPlayer
from threading import Thread

import cv2
import numpy as np
from fastdtw import fastdtw
from scipy.spatial.distance import cosine

import extractor as ex

# Setup basic logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s:%(levelname)s:%(message)s",
)

class VideoAudioPlayer:
    def __init__(self, video_path):
        self.player = MediaPlayer(video_path)
        self.running = True

    def play(self):
        while self.running:
            audio_frame, val = self.player.get_frame()
            if val == 'eof' or not self.running:
                break
            elif audio_frame is None:
                time.sleep(0.01)
            else:
                time.sleep(val)

    def stop(self):
        self.running = False

    def pause(self):
        self.player.toggle_pause()

    def seek(self, timestamp):
        self.player.seek(timestamp, relative=False)

def manage_audio(audio_player, command, param=None):
    if command == 'start':
        Thread(target=audio_player.play).start()
    elif command == 'stop':
        audio_player.stop()
    elif command == 'pause':
        audio_player.pause()
    elif command == 'seek':
        audio_player.seek(param)

            
MATCH_FRAME_ERROR_THRESHOLD = 0.28

def compare_positions(benchmark_video, user_video):
    benchmark_cam = cv2.VideoCapture(benchmark_video)
    user_cam = cv2.VideoCapture(user_video)
    
    display_info = True  # Initialize the flag to True

    if not benchmark_cam.isOpened() or not user_cam.isOpened():
        logging.error("Failed to open one or both videos.")
        return
    
    audio_player = VideoAudioPlayer(benchmark_video)
    manage_audio(audio_player, 'start')

    fps_time = time.time()
    detector_1 = ex.poseDetector(detectionCon=0.7, trackCon=0.7)
    detector_2 = ex.poseDetector(detectionCon=0.7, trackCon=0.7)
    frame_counter = 0
    correct_frames = 0

    # Skip frames to speed up
    skip_frames = 1
    
    while True:
        ret_val, image_1 = user_cam.read()
        # Skip frames for the benchmark video
        for _ in range(skip_frames + 1):  # +1 because we also need to read one frame to process
            ret_val_1, image_2 = benchmark_cam.read()
            if not ret_val_1:
                benchmark_cam.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reset if video ends or fails
                break

        if not ret_val or not ret_val_1:
            user_cam.set(cv2.CAP_PROP_POS_FRAMES, 0)
            benchmark_cam.set(cv2.CAP_PROP_POS_FRAMES, 0)
            manage_audio(audio_player, 'stop')
            continue
        
        # Flip webcam image
        image_1 = cv2.flip(image_1, 1)
        
        image_1, image_2, frame_counter, correct_frames = process_frame(
            image_1,
            image_2,
            detector_1,
            detector_2,
            frame_counter,
            correct_frames,
            fps_time,
        )

        # Concatenate images horizontally
        combined_image = concatenate_images(image_1, image_2)

        # Display the combined image
        cv2.imshow("Game Display", combined_image)
    
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            manage_audio(audio_player, 'stop')
            break
        elif key == ord('s'):
            manage_audio(audio_player, 'pause')
             # Calculate the width for the left half overlay
            overlay_width = combined_image.shape[1] // 2
            # Create the white overlay on the left half of the image
            combined_image[:, :overlay_width] = (255, 255, 255)  # Setting the left half to white
            display_stats(combined_image, frame_counter, correct_frames, 5)  # Display stats for 5 seconds
            # Reset the videos and counters
            manage_audio(audio_player, 'pause')  # Unpause the audio
            manage_audio(audio_player, 'seek', 0)  # Seek to beginning
            benchmark_cam.set(cv2.CAP_PROP_POS_FRAMES, 0)
            user_cam.set(cv2.CAP_PROP_POS_FRAMES, 0)
            frame_counter = 0
            correct_frames = 0
            continue
        elif key == ord('r'):
            manage_audio(audio_player, 'pause')
             # Calculate the width for the left half overlay
            overlay_width = combined_image.shape[1] // 2
            # Create the white overlay on the left half of the image
            combined_image[:, :overlay_width] = (255, 255, 255)  # Setting the left half to white
            # Display a message indicating the pause (optional, but good for user experience)
            display_restart(combined_image, frame_counter, correct_frames, 5)  # Display stats for 20 seconds
            # Reset the videos and counters
            manage_audio(audio_player, 'pause')  # Unpause the audio
            manage_audio(audio_player, 'seek', 0)  # Seek to beginning
            benchmark_cam.set(cv2.CAP_PROP_POS_FRAMES, 0)
            user_cam.set(cv2.CAP_PROP_POS_FRAMES, 0)
            frame_counter = 0
            correct_frames = 0
            continue
        
    benchmark_cam.release()
    user_cam.release()
    cv2.destroyAllWindows()
    # audio_thread.join()  # Ensure audio playback completes
    
def display_restart(image, frame_counter, correct_frames, duration):
    end_time = time.time() + duration
    while time.time() < end_time:
        accuracy = round(100 * correct_frames / max(frame_counter, 1), 2)  # Avoid division by zero
        cv2.putText(image, "Restarting in 5 seconds...", (70, 180), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        cv2.imshow("Game Display", image)
        if cv2.waitKey(1000) & 0xFF == ord('q'):  # Allow exit during display
            break
        
def display_stats(image, frame_counter, correct_frames, duration):
    end_time = time.time() + duration
    while time.time() < end_time:
        accuracy = round(100 * correct_frames / max(frame_counter, 1), 2)  # Avoid division by zero
        cv2.putText(image, f"Dance Steps Accurately Done: {accuracy}%", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        cv2.imshow("Game Display", image)
        if cv2.waitKey(1000) & 0xFF == ord('q'):  # Allow exit during display
            break
    
def concatenate_images(image_1, image_2):
    height = max(image_1.shape[0], image_2.shape[0])
    image_1 = cv2.resize(image_1, (int(image_1.shape[1] * height / image_1.shape[0]), height))
    image_2 = cv2.resize(image_2, (int(image_2.shape[1] * height / image_2.shape[0]), height))
    return np.hstack((image_1, image_2))

def process_frame(
    image_1, image_2, detector_1, detector_2, frame_counter, correct_frames, fps_time
):
    image_1 = cv2.resize(image_1, (720, 640))
    image_2 = cv2.resize(image_2, (720, 640))
    image_1 = detector_1.findPose(image_1)
    lmList_user = detector_1.findPosition(image_1, draw=False)
    image_2 = detector_2.findPose(image_2)
    lmList_benchmark = detector_2.findPosition(image_2, draw=False)

    if (
        not lmList_user
        or not lmList_benchmark
        or len(lmList_user) == 0
        or len(lmList_benchmark) == 0
    ):
        logging.warning("Missing or insufficient landmarks in one or both videos.")
        return image_1, image_2, frame_counter, correct_frames

    try:
        error, _ = fastdtw(lmList_user, lmList_benchmark, dist=cosine)
        image_1 = update_gui(image_1, error, frame_counter, correct_frames, fps_time)
    except Exception as e:
        logging.error(f"Error calculating DTW: {e}")

    return (
        image_1,
        image_2,
        frame_counter + 1,
        correct_frames + (1 if error < MATCH_FRAME_ERROR_THRESHOLD else 0),
    )

def box(text, font_scale, thickness, font, image_1, x, y):
    (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, thickness)
    cv2.rectangle(image_1, (x, y - text_height - 10), (x + text_width, y + 10), (0, 0, 0), -1)
        
def update_gui(image_1, error, frame_counter, correct_frames, fps_time):
    cv2.putText(
        image_1,
        f"Error: {round(100 * error, 2)}%",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 50, 50),
        2,
    )
    if error < 0.3:
        box("CORRECT", 1, 2, cv2.FONT_HERSHEY_SIMPLEX, image_1, 40, 600)
        cv2.putText(
            image_1,
            "CORRECT",
            (40, 600),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )
    else:
        box("INCORRECT", 1, 2, cv2.FONT_HERSHEY_SIMPLEX, image_1, 40, 600)
        cv2.putText(
            image_1,
            "INCORRECT",
            (40, 600),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2,
        )
    cv2.putText(
        image_1,
        f"FPS: {1.0 / (time.time() - fps_time)}",
        (10, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (128, 0, 128),
        2,
    )
    cv2.putText(
        image_1,
        f"Dance Steps Accurately Done: {round(100 * correct_frames / frame_counter, 2)}%",
        (10, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (19, 69, 139),
        2,
    )
    return image_1

def main():
    parser = argparse.ArgumentParser(
        description="Compare user and benchmark video poses."
    )
    parser.add_argument("benchmark_video", help="Path to the benchmark video file")
    args = parser.parse_args()
    
    
    user_video = 0  # assuming webcam
   
    compare_positions(args.benchmark_video, user_video)
    
if __name__ == "__main__":
    main()
