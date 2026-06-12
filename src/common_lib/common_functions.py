from datetime import datetime


def get_time_stamp():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return timestamp
