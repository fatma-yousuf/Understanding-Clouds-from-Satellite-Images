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

model = keras.models.load_model('segment_model1.h5')
