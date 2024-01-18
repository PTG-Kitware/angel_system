from angel_msgs.msg import DialogueUtterance

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
