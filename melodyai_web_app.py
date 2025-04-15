import streamlit as st
import librosa
import soundfile as sf
import numpy as np
import tempfile
from collections import Counter
import math
import matplotlib.pyplot as plt

def detect_key_from_frequencies(f0):
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

def correct_f0_blockwise(audio, sr, f0, target_scale, hop_length=512, tolerance_semitones=0.5, intensity=1.0):
    output = np.copy(audio)
    for i in range(len(f0)):
        freq = f0[i]
        if freq is not None and not math.isnan(freq):
            midi = librosa.hz_to_midi(freq)
            closest = min(target_scale, key=lambda x: abs(x - midi))
            diff = closest - midi
            if abs(diff) > tolerance_semitones:
                new_midi = midi + diff * intensity
                shift_steps = new_midi - midi
                start = i * hop_length
                end = min(len(audio), start + hop_length)
                segment = audio[start:end]
                if len(segment) > 0:
                    shifted = librosa.effects.pitch_shift(segment, sr=sr, n_steps=shift_steps)
                    output[start:end] = shifted[:end - start]
    return output

def correct_pitch_preserving_vibrato(f0, target_scale, tolerance_semitones=0.5, intensity=1.0):
    corrected = []
    for freq in f0:
        if freq is not None and not math.isnan(freq):
            midi_note = librosa.hz_to_midi(freq)
            closest_note = min(target_scale, key=lambda x: abs(x - midi_note))
            diff = closest_note - midi_note
            if abs(diff) > tolerance_semitones:
                new_midi = midi_note + diff * intensity
            else:
                new_midi = midi_note
            corrected_freq = librosa.midi_to_hz(new_midi)
            corrected.append(corrected_freq)
        else:
            corrected.append(None)
    return np.array(corrected)

def plot_pitch(f0_original, f0_corrected, sr, hop_length):
    times = librosa.times_like(f0_original, sr=sr, hop_length=hop_length)
    plt.figure(figsize=(10, 4))
    plt.plot(times, f0_original, label='Original', color='#00b4d8')
    plt.plot(times, f0_corrected, label='Corregido', color='#90e0ef')
    plt.xlabel("Tiempo (s)")
    plt.ylabel("Frecuencia (Hz)")
    plt.title("Pitch detectado vs corregido (por bloques)")
    plt.legend()
    st.pyplot(plt)

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

st.title("🎵 MelodyAI - Afinación por bloques (natural por tramos)")

uploaded_file = st.file_uploader("Sube un archivo de voz (.wav)", type=["wav"])
intensity = st.slider("Nivel de afinación", 0.0, 1.0, 1.0, 0.1)

if uploaded_file is not None:
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name

        st.audio(tmp_path, format="audio/wav")

        audio, sr = librosa.load(tmp_path)
        hop_length = 512
        f0, _, _ = librosa.pyin(audio, fmin=librosa.note_to_hz('C2'),
                                fmax=librosa.note_to_hz('C7'),
                                hop_length=hop_length)

        if all((freq is None or math.isnan(freq)) for freq in f0):
            st.error("⚠️ No se detectó tono. Intenta con otro archivo.")
        else:
            detected_scale_name, detected_scale_notes = detect_key_from_frequencies(f0)
            st.info(f"🎼 Escala detectada automáticamente: {detected_scale_name}")

            corrected_f0 = correct_pitch_preserving_vibrato(f0, detected_scale_notes, intensity=intensity)
            audio_corrected = correct_f0_blockwise(audio, sr, f0, detected_scale_notes, hop_length=hop_length, intensity=intensity)

            st.subheader("🎼 Visualización del pitch")
            plot_pitch(f0, corrected_f0, sr, hop_length)

            output_path = tmp_path.replace(".wav", "_afinado_bloques.wav")
            sf.write(output_path, audio_corrected, sr)
            st.success("✅ Afinación por bloques completada.")
            st.audio(output_path, format="audio/wav")
            with open(output_path, "rb") as f:
                st.download_button("Descargar audio afinado", f, file_name="voz_afinada_bloques.wav")

    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
