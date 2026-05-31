def fuse_emotions(text: dict, face: dict, voice: dict) -> dict:
    sources = [
        (text, 0.45),
        (face, 0.35),
        (voice, 0.20)
    ]

    emotion_scores = {}

    for source, weight in sources:
        emotion = source.get("primary_emotion", "neutral")
        confidence = source.get("confidence", 0.5)
        score = confidence * weight
        emotion_scores[emotion] = emotion_scores.get(emotion, 0) + score

    final_emotion = max(emotion_scores, key=emotion_scores.get)
    final_confidence = round(emotion_scores[final_emotion], 2)

    all_emotions = [s[0].get("primary_emotion") for s in sources]
    agreement = len(set(all_emotions)) == 1

    return {
        "final_emotion": final_emotion,
        "confidence": min(final_confidence + (0.1 if agreement else 0), 0.99),
        "all_agreed": agreement,
        "breakdown": {
            "text": text.get("primary_emotion", "unknown"),
            "face": face.get("primary_emotion", "unknown"),
            "voice": voice.get("primary_emotion", "unknown")
        },
        "scores": emotion_scores
    }