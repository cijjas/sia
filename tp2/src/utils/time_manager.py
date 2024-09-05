
import time
class TimeManager:
    def __init__(self, time_limit):
        self.time_limit = time_limit
        self.start_time = time.time() 

    
    def time_remaining(self):
        elapsed_time = time.time() - self.start_time
        return max(self.time_limit - elapsed_time, 0)

    def time_is_up(self):
        return self.time_remaining() <= 0