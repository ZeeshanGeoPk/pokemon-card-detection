import datetime
import os
import math

import cv2
import numpy as np


class VideoReader:
    def __init__(
        self, video_path: str, frames_per_batch: int = 1, extraction_fps: int = 4
    ):
        """
        A simple wrapper class over OpenCVs cv2.VideoCapture that
        has the ability to return frames in batches as opposed to
        one by one.

        Args:
            video_path: path to the video file
            frames_per_batch: number of frames to return in each call
            extraction_fps: frame rate at which frames will be extracted from
                            video
        """
        self.video_path = video_path
        self.video = cv2.VideoCapture(video_path)
        self.batch_size = frames_per_batch
        self.extraction_fps = extraction_fps
        assert self.batch_size > 0, "Batch size must be at least 1"

    def __len__(self):
        fps_red_ratio = self.extraction_fps / self.fps
        return int(self.video.get(cv2.CAP_PROP_FRAME_COUNT) * fps_red_ratio)
    
    def close(self):
        self.video.release()

    @property
    def n_batches(self):
        return math.ceil(len(self) / self.batch_size)

    @property
    def fps(self):
        return self.video.get(cv2.CAP_PROP_FPS)

    @property
    def frame_size(self):
        return (
            int(self.video.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            int(self.video.get(cv2.CAP_PROP_FRAME_WIDTH)),
        )

    def read(self):
        for i in range(0, len(self), self.batch_size):
            batch = []
            while True:
                success, frame = self.video.read()
                if success:
                    batch.append(frame)
                    if len(batch) == self.batch_size:
                        break
                else:
                    break
            yield batch

    def read_2(self):
        """Read frames keeping in mind extraction fps
        https://stackoverflow.com/a/47632941
        """
        batch, i = [], 0
        while True:
            batch = []
            batch_timestamps = []
            while True:
                # Set frame reader
                success, frame = self.video.read()
                if success:
                    batch.append(frame)
                    # timestamp = milliseconds_to_time(1000 * i * 1 / self.extraction_fps)
                    time_ms = (1000 * i * (1 / self.extraction_fps))
                    batch_timestamps.append(time_ms)
                    i += 1
                    # Set video capture position to millisecons of frame
                    self.video.set(
                        cv2.CAP_PROP_POS_MSEC, (1000 * i * 1 / self.extraction_fps)
                    )
                    if len(batch) >= self.batch_size or len(batch) == len(self):
                        break
                else:
                    return
            yield batch, batch_timestamps

    def get_frame(self, timestamp='00:00:01'):
        hours, minutes, secs = timestamp.split(':')
        total_seconds = int(hours)*60 + int(minutes)*60 + int(secs)
        # Set video capture position to millisecons of frame
        self.video.set(cv2.CAP_PROP_POS_MSEC, total_seconds * 1000)
        success, np_frame = self.video.read()
        if not success:
            return None
        return np_frame


def get_all_videos(directory):
    assert os.path.exists(directory)

    if os.path.isfile(directory):
        return [directory] if directory.endswith(".mp4") else []

    paths = []
    for f in os.listdir(directory):
        paths += get_all_videos(os.path.join(directory, f))

    return paths


def get_videos_from_file(videos_dir, videos_file, video_ids):
    with open(videos_file) as f:
        video_names = f.read().split("\n")
    paths = [os.path.join(videos_dir, vn) for vn in video_names if vn]

    if video_ids is not None:
        start_id, end_id = video_ids
        paths = paths[start_id:end_id]
    return paths


def create_video_from_images(images, fps, save_path):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    if isinstance(images[0], np.ndarray):
        h, w, c = images[0].shape
    else:
        w, h = images[0].size
    writer = cv2.VideoWriter(save_path, fourcc, fps, (w, h))
    for image in images:
        writer.write(image)
    writer.release()


def crop_video(video_path: str, start_time: float, end_time: float, save_path: str):
    # Start reading video
    video = cv2.VideoCapture(video_path)

    # Move to start time
    video.set(cv2.CAP_PROP_POS_MSEC, float(start_time * 1000))

    # Read frames upto end time
    fps = video.get(cv2.CAP_PROP_FPS)
    n_frames_till_end = int((end_time - start_time) * fps)
    frames = [video.read()[1] for _ in range(n_frames_till_end)]
    video.release()

    # Write video
    create_video_from_images(frames, fps, save_path)
    
    
class VideoWriter:
    def __init__(self, output_path, fps, size):
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        h, w = size
        self.writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
    
    def add_frame(self, image):
        self.writer.write(image)
    
    def close(self):
        self.writer.release()
        
        
def milliseconds_to_time(ms):
    s, ms = divmod(ms, 1000)
    m, s = divmod(s, 60)
    h, m = divmod(m, 60)
    return datetime.time(hour=int(h), minute=int(m), second=int(s), microsecond=ms*1000)