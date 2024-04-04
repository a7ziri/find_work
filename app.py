import streamlit as st
import pandas as pd
import torch
import numpy as np
from sentence_transformers import SentenceTransformer , util
import speech_recognition
from PIL import Image
recognizer = speech_recognition.Recognizer()
microphone = speech_recognition.Microphone()
def record_and_recognize_audio(*args: tuple):
    """
    Запись и распознавание аудио
    """
    with microphone:
        recognized_data = ""

        # регулирование уровня окружающего шума
        recognizer.adjust_for_ambient_noise(microphone, duration=2)

        try:
            print("Listening...")
            audio = recognizer.listen(microphone, 5, 5)

        except speech_recognition.WaitTimeoutError:
            print("Can you check if your microphone is on, please?")
            return

        # использование online-распознавания через Google 
        try:
            print("Started recognition...")
            recognized_data = recognizer.recognize_google(audio, language="ru").lower()

        except speech_recognition.UnknownValueError:
            pass

        # в случае проблем с доступом в Интернет происходит выброс ошибки
        except speech_recognition.RequestError:
            print("Check your Internet Connection, please")

        return recognized_data

def _cosine(e1, e2):
    cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
    e1 = torch.from_numpy(e1)
    e2 = torch.from_numpy(e2)
    return cos(e1 , e2)


@st.cache_resource
def load_model():
    data = pd.read_csv('data/IT_vacancies_full.csv')
    model =  SentenceTransformer('intfloat/multilingual-e5-large-instruct')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return model, device , data

def main():
    if 'result_index' not in st.session_state:
        st.session_state.result_index = 0
    logo = Image.open("logo.jpg")
    st.sidebar.image(logo, use_column_width=True)
    st.sidebar.link_button('git' , 'https://github.com/a7ziri/find_work')
    st.sidebar.link_button('socials' , 'https://t.me/assskelad')
    features =np.load('models/emb/emb_for_model_e5.npy')
    model, device , data = load_model()
    text_user = st.text_input("Введите текст")
    show_more_option = st.checkbox('показать ещё вариант')
    if st.button('голосовой ввод'):
       
        voice_input = record_and_recognize_audio()
        text_user =  voice_input
        st.write("Введенный текст:", text_user)
    # elif:
    #      st.write("Не удалось распознать речь")


    if  len(text_user) > 5:
        embeddings = model.encode(text_user)
        with torch.no_grad():
            cos_scores = _cosine(embeddings, features)
            top_results = torch.topk(cos_scores, k=5)
            indices = [i.item() for i in top_results.indices]
            if  not(show_more_option) :
                ids = indices[st.session_state.result_index]
                st.session_state.result_text = f"Вот подходящий вариант: {data['Name'][ids]}"
                st.session_state.description = data["Description"][ids]
                st.session_state.experience = data["Experience"][ids]
                st.write(f"### {st.session_state.result_text}")
                st.write(st.session_state.description)
                st.write(st.session_state.experience)

            if show_more_option:
                if st.session_state.result_index + 1 < len(indices):
                    st.session_state.result_index += 1
                    ids = indices[st.session_state.result_index]
                    st.session_state.result_text = f"Вот подходящий вариант: {data['Name'][ids]}"
                    st.session_state.description = data["Description"][ids]
                    st.session_state.experience = data["Experience"][ids]

                    st.write(f"### {st.session_state.result_text}")
                    st.write(st.session_state.description)
                    st.write(st.session_state.experience)
                else:
                    st.write("Больше вариантов нет")
if __name__ == "__main__":
    main()