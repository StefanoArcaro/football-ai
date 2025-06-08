from threading import Lock


class SharedState:
    def __init__(self):
        self.latest_frame = None
        self.latest_stats = {}
        self.lock = Lock()

    def update(self, frame, stats):
        with self.lock:
            self.latest_frame = frame
            self.latest_stats = stats

    def get(self):
        with self.lock:
            return self.latest_frame, self.latest_stats


shared_state = SharedState()
