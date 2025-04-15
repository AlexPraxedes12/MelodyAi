
import streamlit as st
import librosa
import soundfile as sf
import numpy as np
import tempfile
from collections import Counter
import math

def detect_key_from_frequencies(f0):
    # Convertir solo si no es None ni NaN
    notes = [round(librosa.hz_to_midi(freq)) for freq in f0 if freq is not None and not math.isnan(freq)]
    if len(notes) < 3:
        raise ValueError("No se detectaron suficientes notas para identificar una escala.")
    note_counts = Counter(notes)
    most_common_notes = [note for note, count in note_counts.most_common(7)]
    SCALES = {
        "Do Mayor (C Major)": [60, 62, 64, 65, 67, 69, 71],
        "Re Mayor (D Major)": [62, 64, 66, 67, 69, 71, 73],
        "Mi Menor (E Minor)": [64, 66, 67, 69, 71, 72, 74],
        "La Menor (A Minor)": [69, 71, 72, 74, 76, 77, 79]
    }
    best_match = max(SCALES.items(), key=lambda item: len(set(item[1]) & set(most_common_notes)))
    return best_match

def correct_pitch_to_scale(f0, target_scale, intensity=1.0):
    corrected = []
    for freq in f0:
        if freq is not None and not math.isnan(freq):
            midi_note = librosa.hz_to_midi(freq)
            closest_note = min(target_scale, key=lambda x: abs(x - midi_note))
            new_midi = midi_note + (closest_note - midi_note) * intensity
            corrected_freq = librosa.midi_to_hz(new_midi)
            corrected.append(corrected_freq)
        else:
            corrected.append(None)
    return np.array(corrected)

def apply_pitch_shift(audio, sr, f0_original, f0_corrected, hop_length=512):
    audio_shifted = np.copy(audio)
    for i in range(0, len(audio), hop_length):
        frame_idx = i // hop_length
        if frame_idx >= len(f0_original) or f0_original[frame_idx] is None or f0_corrected[frame_idx] is None:
            continue
        pitch_ratio = f0_corrected[frame_idx] / f0_original[frame_idx]
        segment = audio[i:i+hop_length]
        audio_shifted[i:i+hop_length] = librosa.effects.pitch_shift(audio[i:i+hop_length], sr, np.log2(pitch_ratio) * 12)
        audio_shifted[i:i+len(segment_shifted)] = segment_shifted[:len(segment)]
    return audio_shifted

st.markdown("""
    <style>
    .reportview-container {
        background-color: #0d1b2a;
        color: #e0e1dd;
    }
    .sidebar .sidebar-content {
        background-color: #1b263b;
    }
    h1 {
        color: #00b4d8;
    }
    .stButton>button {
        background-color: #00b4d8;
        color: white;
        border-radius: 8px;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🎵 MelodyAI - Afinación automática con detección de escala + control de intensidad")

uploaded_file = st.file_uploader("Sube un archivo de voz (.wav)", type=["wav"])
intensity = st.slider("Nivel de afinación", 0.0, 1.0, 1.0, 0.1, help="0 = sin afinación, 1 = máxima precisión")

if uploaded_file is not None:
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name

        st.audio(tmp_path, format="audio/wav")

        audio, sr = librosa.load(tmp_path)
        f0, voiced_flag, voiced_prob = librosa.pyin(
            audio, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7')
        )

        if all((freq is None or math.isnan(freq)) for freq in f0):
            st.error("⚠️ No se detectó tono en el archivo. Intenta con una grabación más clara o más larga.")
        else:
            detected_scale_name, detected_scale_notes = detect_key_from_frequencies(f0)
            st.info(f"🎼 Escala detectada automáticamente: {detected_scale_name}")

            corrected_f0 = correct_pitch_to_scale(f0, detected_scale_notes, intensity=intensity)
            audio_corrected = apply_pitch_shift(audio, sr, f0, corrected_f0)

            output_path = tmp_path.replace(".wav", "_melodyai_output.wav")
            sf.write(output_path, audio_corrected, sr)

            st.success("✅ Afinación completada.")
            st.audio(output_path, format="audio/wav")
            with open(output_path, "rb") as f:
                st.download_button("Descargar audio afinado", f, file_name="voz_afinada_melodyai.wav")

    except Exception as e:
        st.error(f"❌ Ocurrió un error procesando el archivo: {str(e)}")
