
import time
class TimeManager:
    def __init__(self, time_limit):
        self.time_limit = time_limit
        self.start_time = time.time() # time measure in seconds
        print(f"Time limit: {self.time_limit}")

    
    def time_remaining(self):
        elapsed_time = time.time() - self.start_time
        return max(self.time_limit - elapsed_time, 0)

    def time_is_up(self):
        time_remaining = self.time_remaining()
        if time_remaining <= 0:
            print("Time is up!")
            return True
        return False