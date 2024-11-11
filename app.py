import streamlit as st
import os
import base64
import torch
import torchaudio
from audiocraft.models import MusicGen

# Hide Streamlit Header and Footer
hide = """
<style>
    #root > div:nth-child(1) > div.withScreencast > div > div > header,
    footer {visibility: hidden;}
</style>
"""
st.markdown(hide, unsafe_allow_html=True)

@st.cache_resource
def load_model():
    model = MusicGen.get_pretrained("facebook/musicgen-small")
    return model

def generate_music_tensor(description, duration: int):
    model = load_model()
    model.set_generation_params(
        use_sampling=True,
        top_k=250,
        duration=duration
    )
    output = model.generate(
        descriptions=[description],
        progress=True,
        return_tokens=True
    )
    return output[0]

def save_audio(samples: torch.Tensor):
    sample_rate = 32000
    save_path = "audio_output/"
    os.makedirs(save_path, exist_ok=True)

    samples = samples.detach().cpu()
    if samples.dim() == 2:
        samples = samples[None, ...]

    for idx, audio in enumerate(samples):
        audio_path = os.path.join(save_path, f'audio_{idx}.wav')
        try:
            torchaudio.save(audio_path, audio, sample_rate, backend='soundfile')
            print(f"Saved audio to {audio_path}")
        except Exception as e:
            print(f"Failed to save audio file: {e}")

def get_binary_file_downloader_html(bin_file, file_label='File'):
    with open(bin_file, 'rb') as f:
        data = f.read()
        bin_str = base64.b64encode(data).decode()
        href = f'<a href="data:application/octet-stream;base64,{bin_str}" download="{os.path.basename(bin_file)}">Download {file_label}</a>'
        return href

def add_custom_background():
    video_file_path = "background.mp4"
    video_base64 = get_base64_video(video_file_path)

    st.markdown(
        f"""
        <style>
        .stApp {{
            background-color: transparent;
            overflow: hidden;
        }}
        #background-video {{
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            object-fit: cover;
            z-index: -1;
        }}
        .content {{
            position: relative;
            z-index: 1;
            padding: 20px;
        }}
        </style>
        <video autoplay muted loop id="background-video">
            <source src="data:video/mp4;base64,{video_base64}" type="video/mp4">
        </video>
        """,
        unsafe_allow_html=True
    )

def get_base64_video(video_file):
    with open(video_file, "rb") as video_file:
        video_bytes = video_file.read()
        encoded_video = base64.b64encode(video_bytes).decode()
    return encoded_video

def main():
    add_custom_background()
    st.markdown('<div class="content">', unsafe_allow_html=True)
    st.title("Text to Music Generation Project By Shirish and Charan")

    with st.expander("See Explanation"):
        st.write('''Our team created this music-generating app utilizing Meta's Audiocraft library. 
                    It follows the MusicGen model. It can generate music based on the natural language description you provide.''')

    text_area = st.text_area("Enter your description...")
    time_slider = st.slider("Select time duration (in seconds)", 2, 60, 5)

    if text_area and time_slider:
        st.json({
            "Your description": text_area,
            "Selected time duration (in seconds)": time_slider
        })

        st.subheader("Generated Music!")
        music_tensors = generate_music_tensor(text_area, time_slider)
        save_audio(music_tensors)

        audio_file_path = 'audio_output/audio_0.wav'
        if os.path.exists(audio_file_path):
            with open(audio_file_path, 'rb') as audio_file:
                audio_bytes = audio_file.read()
                st.audio(audio_bytes, format='audio/wav')
                st.markdown(get_binary_file_downloader_html(audio_file_path, 'Audio'), unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
