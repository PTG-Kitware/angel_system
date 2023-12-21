from angel_msgs.msg import DialogueUtterance

def get_intent_or(msg: DialogueUtterance,
                  or_value: str = "not available") -> str:
    """
    Returns the msg intent classification information. If the value is absent,
    the or_value is passed in.
    """
    return msg.intent if msg.intent else or_value

def get_emotion_or(msg: DialogueUtterance,
                   or_value: str = "not available") -> str:
    """
    Returns the msg emotion classification information. If the value is absent,
    the or_value is passed in.
    """
    return msg.emotion if msg.emotion else or_value

def copy_dialogue_utterance(msg: DialogueUtterance,
                            node_name,
                            copy_time) -> DialogueUtterance:
    msg = DialogueUtterance()
    msg.header.frame_id = node_name
    msg.utterance_text = msg.utterance_text

    # Assign new time for publication.
    msg.header.stamp = copy_time

    # Copy over intent classification information if present.
    if msg.intent:
        msg.intent = msg.intent
        msg.intent_confidence_score = msg.intent_confidence_score

    # Copy over intent classification information if present.
    if msg.emotion:
        msg.emotion = msg.emotion
        msg.emotion_confidence_score = msg.emotion_confidence_score

    return msg
