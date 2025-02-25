import streamlit as st
import requests
from PIL import Image
from io import BytesIO
from transformers import BlipProcessor, BlipForConditionalGeneration, GPT2Tokenizer, GPT2LMHeadModel
from gtts import gTTS
import torch

# Load the BLIP model for image captioning
caption_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
caption_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

def generate_caption(image):
    # Preprocess the image
    inputs = caption_processor(images=image, return_tensors="pt")

    # Generate caption
    out = caption_model.generate(**inputs)
    caption = caption_processor.decode(out[0], skip_special_tokens=True)
    return caption


def text_to_speech(text, filename="output.mp3"):
    tts = gTTS(text=text, lang='en')
    tts.save(filename)
    return filename

# Streamlit app
st.title("Image Captioning with TTS")

# Upload an image
uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    image = Image.open(uploaded_image).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Generate caption
    caption = generate_caption(image)
    st.write(f"**Generated Caption:** {caption}")

    # Convert caption to speech
    audio_file = text_to_speech(caption)
    st.audio(audio_file, format='audio/mp3', start_time=0)