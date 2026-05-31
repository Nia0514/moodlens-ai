import librosa
import numpy as np
import io

def analyze_voice(audio_bytes: bytes) -> dict:
    try:
        audio, sr = librosa.load(io.BytesIO(audio_bytes), sr=None)

        pitches, magnitudes = librosa.piptrack(y=audio, sr=sr)
        avg_pitch = float(np.mean(pitches[pitches > 0])) if np.any(pitches > 0) else 0

        tempo, _ = librosa.beat.beat_track(y=audio, sr=sr)
        energy = float(np.mean(librosa.feature.rms(y=audio)))

        if avg_pitch > 200 and energy > 0.05:
            emotion = "excited"
        elif avg_pitch < 120 and energy < 0.02:
            emotion = "sad"
        elif float(tempo) > 140 and energy > 0.06:
            emotion = "angry"
        elif energy < 0.015:
            emotion = "calm"
        else:
            emotion = "neutral"

        return {
            "primary_emotion": emotion,
            "confidence": round(min(energy * 10 + 0.4, 0.95), 2),
            "features": {
                "pitch": round(avg_pitch, 2),
                "tempo": round(float(tempo), 2),
                "energy": round(energy, 4)
            }
        }

    except Exception as e:
        return {
            "primary_emotion": "neutral",
            "confidence": 0.4,
            "error": str(e)
        }