import os
import torch
import torchaudio
import wave
import streamlit as st
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

# ✅ Streamlit UI
st.title("🎤 Speech-to-Text Transcription with Wav2Vec2")
uploaded_file = st.file_uploader("Upload a WAV file", type=["wav"])

if uploaded_file is not None:
    # Save to temporary file
    file_path = "temp_audio.wav"
    with open(file_path, "wb") as f:
        f.write(uploaded_file.read())

    # ✅ Validate WAV file
    try:
        with wave.open(file_path, 'rb') as wf:
            channels = wf.getnchannels()
            sample_width = wf.getsampwidth()
            frame_rate = wf.getframerate()
            st.success(f"WAV File Info: Channels={channels}, Sample Width={sample_width}, Frame Rate={frame_rate}")
    except wave.Error as e:
        st.error(f"Wave Error: {e}")
        st.stop()

    # ✅ Load audio
    try:
        waveform, sample_rate = torchaudio.load(file_path)
        st.success("Audio loaded successfully!")
    except Exception as e:
        st.error(f"Error loading audio: {e}")
        st.stop()

    # ✅ Load pre-trained model
    st.info("Loading pre-trained Wav2Vec2 model...")
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
    model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")
    model.eval()

    # ✅ Resample if needed
    target_sample_rate = 16000
    if sample_rate != target_sample_rate:
        st.warning(f"Resampling from {sample_rate} Hz to {target_sample_rate} Hz...")
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=target_sample_rate)
        waveform = resampler(waveform)
    else:
        st.success("Sample rate is already 16000 Hz.")

    # ✅ Transcribe
    st.info("Transcribing audio...")
    input_values = processor(waveform.squeeze().numpy(), return_tensors="pt", sampling_rate=target_sample_rate).input_values
    with torch.no_grad():
        logits = model(input_values).logits
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = processor.decode(predicted_ids[0])

    # ✅ Output transcription
    st.subheader("📝 Transcription Result:")
    st.text(transcription)
