import streamlit as st
from PIL import Image
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain import PromptTemplate
from google.colab import drive
import os
from langchain.llms import CTransformers
from langchain.embeddings import HuggingFaceEmbeddings
from tensorflow import keras
import numpy as np
from langchain.chains import RetrievalQA
import streamlit as st
from langchain.llms import HuggingFaceHub
from streamlit_extras.add_vertical_space import add_vertical_space
import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

css = '''
<style>
.chat-message {
    padding: 1.5rem; border-radius: 0.5rem; margin-bottom: 1rem; display: flex
}
.chat-message.user {
    background-color: #2b313e
}
.chat-message.bot {
    background-color: #475063
}
.chat-message .avatar {
  width: 20%;
}
.chat-message .avatar img {
  max-width: 78px;
  max-height: 78px;
  border-radius: 50%;
  object-fit: cover;
}
.chat-message .message {
  width: 80%;
  padding: 0 1.5rem;
  color: #fff;
}
'''

model = keras.models.load_model('segment_model1.h5')


with st.sidebar:
    st.title('🤗 Understanding Cloud')
    add_vertical_space(5)

if uploaded_file is not None:
    st.image(uploaded_file, caption='Uploaded Photo', use_column_width=True,width = 100)
    image = Image.open(uploaded_file)
    input_image = np.array(image.convert('L').resize((320, 480)))
    print("image resized shape: ", input_image.shape)
    input_image = input_image / 255.0
    print("image expanded shape", np.expand_dims(input_image, axis=0).shape)
    if selected_option == "Knee X-ray":
      predictions = model_knee.predict(np.expand_dims(input_image, axis=0))
      class_index = np.argmax(predictions)
      st.write(diagnose_health_knee(class_index))


    elif selected_option == "Chest X-ray":
      predictions = model.predict(np.expand_dims(input_image, axis=0))
      class_index = np.argmax(predictions)

      st.write(diagnose_health(class_index))
    st.write(f'Confidence: {predictions[0][class_index]:.2f}')


    query = st.text_input("Ask questions about your Xray:")
