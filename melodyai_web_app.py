
import streamlit as st
import librosa
import soundfile as sf
import numpy as np
import tempfile

# Diccionario de escalas básicas (mayores y menores)
SCALES = {
    "Do Mayor (C Major)": [60, 62, 64, 65, 67, 69, 71],
    "Re Mayor (D Major)": [62, 64, 66, 67, 69, 71, 73],
    "Mi Menor (E Minor)": [64, 66, 67, 69, 71, 72, 74],
    "La Menor (A Minor)": [69, 71, 72, 74, 76, 77, 79]
}

def correct_pitch_to_scale(f0, target_scale):
    corrected = []
    for freq in f0:
        if freq is not None:
            midi_note = librosa.hz_to_midi(freq)
            closest_note = min(target_scale, key=lambda x: abs(x - midi_note))
            corrected_freq = librosa.midi_to_hz(closest_note)
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
        segment_shifted = librosa.effects.pitch_shift(segment, sr, n_steps=np.log2(pitch_ratio) * 12)
        audio_shifted[i:i+len(segment_shifted)] = segment_shifted[:len(segment)]
    return audio_shifted

# Estilo visual
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

st.title("🎵 MelodyAI - Afinación automática de voz")

uploaded_file = st.file_uploader("Sube un archivo de voz (.wav)", type=["wav"])
selected_scale = st.selectbox("Selecciona una escala musical:", list(SCALES.keys()))

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    st.audio(tmp_path, format="audio/wav")

    audio, sr = librosa.load(tmp_path)
    f0, voiced_flag, voiced_prob = librosa.pyin(audio, 
                                                 fmin=librosa.note_to_hz('C2'), 
                                                 fmax=librosa.note_to_hz('C7'))

    target_scale = SCALES[selected_scale]
    corrected_f0 = correct_pitch_to_scale(f0, target_scale)
    audio_corrected = apply_pitch_shift(audio, sr, f0, corrected_f0)

    output_path = tmp_path.replace(".wav", "_melodyai_output.wav")
    sf.write(output_path, audio_corrected, sr)

    st.success("✅ Afinación completada en la escala seleccionada.")
    st.audio(output_path, format="audio/wav")

    with open(output_path, "rb") as f:
        st.download_button("Descargar audio afinado", f, file_name="voz_afinada_melodyai.wav")
