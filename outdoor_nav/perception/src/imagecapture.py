import pyrealsense2 as rs
import threading
import numpy as np

class ImageCaptureAsync:
    def __init__(self, width=640, height=480, fps=15):
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
        config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        self.pipeline.start(config)
        self.frames = self.pipeline.wait_for_frames()
        self.started = False
        self.read_lock = threading.Lock()

    def start(self):
        if self.started:
            print('[!] Asynchronous video capturing has already been started.')
            return None
        self.started = True
        self.thread = threading.Thread(target=self.update, args=())
        self.thread.start()
        return self

    def update(self):
        while self.started:
            frames = self.pipeline.wait_for_frames()
            with self.read_lock:
                self.frames = frames

    def read(self):
        with self.read_lock:
            depth_frame = self.frames.get_depth_frame()
            color_frame = self.frames.get_color_frame()
        return np.asanyarray(depth_frame.get_data()), np.asanyarray(color_frame.get_data())

    def stop(self):
        self.started = False
        self.thread.join()

    def __exit__(self, exec_type, exc_value, traceback):
        self.pipeline.stop()
