import time

class Timer:
    def __init__(self, name="Timer"):
        self.name = name
        self.start_time = None
        self.end_time = None

    def start(self):
        self.start_time = time.time()

    def stop(self):
        self.end_time = time.time()

    def report(self):
        if self.start_time and self.end_time:
            print(f"[{self.name}] Thời gian: {self.end_time - self.start_time:.2f} giây")
