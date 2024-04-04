import streamlit as st
import pandas as pd
import torch
import transformers
from transformers import AutoTokenizer , AutoModel
import numpy as np
import hydra
from omegaconf import DictConfig
import speech_recognition as sr

def listen():
    voice_recognizer = sr.Recognizer()
    pause_threshold = 0.03
    with sr.Microphone() as source:
        audio = voice_recognizer.listen(source, timeout = 5, phrase_time_limit = 30)
    try:
        voice_text = voice_recognizer.recognize_google(audio, language="ru")
        return voice_text
    except sr.UnknownValueError:
        return "Ошибка распознания"
    except sr.RequestError:
        return "Ошибка соединения"


def _cosine(e1, e2):
    cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
    e1 = torch.from_numpy(e1)
    e2 = torch.from_numpy(e2)
    return cos(e1 , e2)


def load_model():
    data = pd.read_csv('data/data1.csv')
    tokenizer = AutoTokenizer.from_pretrained('DeepPavlov/rubert-base-cased')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = AutoModel.from_pretrained('DeepPavlov/rubert-base-cased').to(device)
    return tokenizer, model, device , data

def main():

    features =np.load('models/emb/emb_for_model_deep.npy')
    tokenizer, model, device , data = load_model()
    text_user = st.text_input("Введите текст")
    if st.button('голосовой ввод'):
        text_user =  listen()
        st.write("Введенный текст:", text_user)
    
   
    text_for_example_tok = tokenizer(
    text_user, max_length=512, return_tensors="pt", padding=True, truncation=True
    )
    with torch.no_grad():
        emb = model(text_for_example_tok["input_ids"].to(device) ,text_for_example_tok["attention_mask"].to(device))
        fech_demo = emb.last_hidden_state[:, 0, :].cpu().numpy()
    data['sim'] = _cosine(fech_demo , features)
    if len(text_user) > 5:
        st.write((data[data['sim'] == max(data['sim'])]))
       


if __name__ == "__main__":
    main()