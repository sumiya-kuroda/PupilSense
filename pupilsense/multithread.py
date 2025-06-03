import time
import cv2
import threading
import queue


class CapMultiThreading:
    def __init__(self, cap_id, maxsize=512):
        # Assert if cap_id is not inserted
        assert cap_id is not None, "Please specify cap"
        self.frame_count = 0
        self.frame_queue = queue.Queue(maxsize=maxsize)
        self.cap = None
        self.cap_id = cap_id

        self.ended = False  # We want to know if the video has ended
        self.start()

    def start(self):
        self.cap = cv2.VideoCapture(self.cap_id)
        threading.Thread(target=self.update, args=()).start()
        return self

    def update(self):
        while True:
            if not self.frame_queue.full():
                ret, frame = self.cap.read()
                if not ret:
                    self.ended = True
                    break
                self.frame_queue.put([ret, frame])
                self.frame_count += 1
            else:
                time.sleep(0.01)
            # If program is closed, release the cap
            if self.ended:
                break
        self.cap.release()
        print("CapMultiThreading: Cap released")

    def get_frame(self):
        if self.frame_queue.empty() and self.ended:
            return False, None
        ret, frame = self.frame_queue.get()
        return ret, frame

    def release(self):
        self.ended = True

    def get_totalframe(self):
        return self.cap.get(cv2.CAP_PROP_FRAME_COUNT)