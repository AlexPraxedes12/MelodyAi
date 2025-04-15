
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

def compute_average_shift(f0_original, f0_corrected):
    deltas = []
    for o, c in zip(f0_original, f0_corrected):
        if o is not None and c is not None and not math.isnan(o) and not math.isnan(c):
            semitone_shift = 12 * np.log2(c / o)
            deltas.append(semitone_shift)
    if len(deltas) == 0:
        return 0.0
    return np.mean(deltas)

def plot_pitch(f0_original, f0_corrected, sr, hop_length):
    times = librosa.times_like(f0_original, sr=sr, hop_length=hop_length)
    plt.figure(figsize=(10, 4))
    plt.plot(times, f0_original, label='Original', color='#00b4d8')
    plt.plot(times, f0_corrected, label='Corregido', color='#90e0ef')
    plt.xlabel("Tiempo (s)")
    plt.ylabel("Frecuencia (Hz)")
    plt.title("Pitch detectado vs corregido")
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

st.title("🎵 MelodyAI - Afinación + Detección de Escala + Visualización de Notas")

uploaded_file = st.file_uploader("Sube un archivo de voz (.wav)", type=["wav"])
intensity = st.slider("Nivel de afinación", 0.0, 1.0, 1.0, 0.1, help="0 = sin afinación, 1 = máxima precisión")

if uploaded_file is not None:
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name

        st.audio(tmp_path, format="audio/wav")

        audio, sr = librosa.load(tmp_path)
        hop_length = 512
        f0, voiced_flag, voiced_prob = librosa.pyin(
            audio, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'), hop_length=hop_length
        )

        if all((freq is None or math.isnan(freq)) for freq in f0):
            st.error("⚠️ No se detectó tono en el archivo. Intenta con una grabación más clara o más larga.")
        else:
            detected_scale_name, detected_scale_notes = detect_key_from_frequencies(f0)
            st.info(f"🎼 Escala detectada automáticamente: {detected_scale_name}")

            corrected_f0 = correct_pitch_to_scale(f0, detected_scale_notes, intensity=intensity)
            average_shift = compute_average_shift(f0, corrected_f0)
            audio_corrected = librosa.effects.pitch_shift(audio, sr=sr, n_steps=average_shift)

            # Mostrar gráfico de notas
            st.subheader("🎼 Visualización del pitch")
            plot_pitch(f0, corrected_f0, sr, hop_length)

            output_path = tmp_path.replace(".wav", "_melodyai_output.wav")
            sf.write(output_path, audio_corrected, sr)

            st.success("✅ Afinación completada.")
            st.audio(output_path, format="audio/wav")
            with open(output_path, "rb") as f:
                st.download_button("Descargar audio afinado", f, file_name="voz_afinada_melodyai.wav")

    except Exception as e:
        st.error(f"❌ Ocurrió un error procesando el archivo: {str(e)}")
