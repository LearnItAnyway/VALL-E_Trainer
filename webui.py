import argparse
import logging
import os
import pathlib
import time
import tempfile
import platform
import webbrowser
import sys
import torch, torchaudio
import random
import numpy as np
import gradio as gr

from models.utils import (
    AudioTokenizer, AttributeDict,
    TextTokenizer, get_text_token_collater
)
from models import get_model

from vocos import Vocos
from encodec.utils import convert_audio

parser = argparse.ArgumentParser(description='Audio-Text Model Training')
parser.add_argument('--model_path', type=str, default='vall-e_ko_v1.pt')
parser.add_argument('--token_path', type=str, default='unique_text_tokens_v1.k2symbols')
args = parser.parse_args()

language = 'en'
if args.model_path.find('ko') >=0:
    language = 'ko' 
text_tokenizer = TextTokenizer(language=language)
device = torch.device("cpu")
if torch.cuda.is_available():
    device = torch.device("cuda", 0)

checkpoint = torch.load(args.model_path, map_location='cpu')
model = get_model(AttributeDict(checkpoint))
#model.prefix_mode = 0
missing_keys, unexpected_keys = model.load_state_dict(
    checkpoint["model"], strict=True
)
assert not missing_keys

model.eval()
model.to(device)
text_collater = get_text_token_collater(args.token_path)

# Encodec model
audio_tokenizer = AudioTokenizer(device)

# Vocos decoder
vocos = Vocos.from_pretrained('charactr/vocos-encodec-24khz').to(device)

model.to(device)

def split_text_into_chunks(text, max_words=20):
    # Function to split text into sentences
    def split_into_sentences(text):
        sentences, sentence = [], []
        for word in text.split():
            sentence.append(word)
            if word.endswith(('.','!','?')):
                sentences.append(' '.join(sentence))
                sentence = []
        if sentence:
            sentences.append(' '.join(sentence))
        return sentences

    # Function to split a sentence into chunks
    def split_sentence_into_chunks(sentence, max_words):
        words = sentence.split()
        return [' '.join(words[i:i+max_words]) for i in range(0, len(words), max_words)]

    # Split text into sentences and then into chunks
    sentences = split_into_sentences(text)
    chunks = []
    for sentence in sentences:
        chunks.extend(split_sentence_into_chunks(sentence, max_words))
    return chunks

@torch.no_grad()
def infer_from_prompt(text_prompt, audio_prompt, text):
    # Split text into chunks
    text_chunks = split_text_into_chunks(text, max_words=20)
    
    combined_audio = None
    for chunk in text_chunks:
        # Process each chunk to generate audio
        text_tokens, text_tokens_lens = text_collater(
            [text_tokenizer(f"{text_prompt} {chunk}".strip())]
        )
        _, enroll_x_lens = text_collater(
            [text_tokenizer(text=f"{text_prompt}".strip())]
        )

        wav_pr, sr = torchaudio.load(audio_prompt)
        wav_pr = convert_audio(wav_pr, sr, audio_tokenizer.sample_rate, audio_tokenizer.channels)
        audio_prompts = audio_tokenizer.encode(wav_pr.unsqueeze(0))[0][0].transpose(2, 1).to(device)

        encoded_frames = model.inference(
            text_tokens.to(device), text_tokens_lens.to(device),
            audio_prompts, enroll_x_lens=enroll_x_lens,
            top_k=-100, temperature=1)
        vocos_features = vocos.codes_to_features(encoded_frames.permute(2, 0, 1))
        samples = vocos.decode(vocos_features, bandwidth_id=torch.tensor([2], device=device))

        # Concatenate the audio
        if combined_audio is None:
            combined_audio = samples
        else:
            combined_audio = torch.cat((combined_audio, samples), dim=1)

    message = f"Sythesized text: {text}"
    return message, (24000, combined_audio.squeeze(0).cpu().numpy())

# Rest of the code remains the same


app = gr.Blocks(title="VALL-E Demo")
with app:
    #gr.Markdown(top_md)
    with gr.Tab("VALL-E Demo"):
        with gr.Row():
            with gr.Column():
                text_prompt = gr.TextArea(label="Input Text",
                                      placeholder="Type text in the audio file (Korean)",)
                audio_prompt= gr.Audio(label="Input Audio", source='upload', interactive=True, type="filepath")
                text_input = gr.TextArea(label="Output Text",
                                      placeholder="Type text you want to generate (Korean)",)
            with gr.Column():
                text_output = gr.Textbox(label="Message")
                audio_output= gr.Audio(label="Output Audio")
                btn = gr.Button("Generate!")
                btn.click(infer_from_prompt,
                          inputs=[text_prompt, audio_prompt, text_input],
                          outputs=[text_output, audio_output])

webbrowser.open("http://127.0.0.1:7860")
app.launch(share=True)
