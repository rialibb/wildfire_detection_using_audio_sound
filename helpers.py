#Sampling Rate, in Hertz
SAMPLING_RATE = 44100

def duration_to_frame_numbers(duration: int) -> int:
    """Returns the number of frames in a samlpe of the given duration
    
    Args:
        duration: the duration, in ms

    Returns:
        the number of frames
    """
    return duration*SAMPLING_RATE//1000