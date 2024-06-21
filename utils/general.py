from typing import Dict

def get_emotion_counter(is_filter: bool) -> Dict:
    """
    Returns a dictionary of emotions with their initial count.

    Parameters
    ----------
    is_filter : bool
        If True, the dictionary will contain only the emotions that are in the filter.
        If False, the dictionary will contain all the emotions.

    Returns
    -------
    Dict
        A dictionary of emotions with their initial count.
    """
    if is_filter:
        emotion_counter = {"Positive": 0, 
                           "Negative": 0, 
                           "Neutral": 0}
    else:
        emotion_counter = {"angry": 0,
                           "disgust": 0,
                           "fear": 0,
                           "happy": 0,
                           "sad": 0,
                           "surprise": 0,
                           "neutral": 0}

    return emotion_counter
