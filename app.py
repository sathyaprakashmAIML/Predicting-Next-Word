import numpy as np 
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
import streamlit as st
model=load_model('model.weights.h5')
st.title("Shakesspeare-Hamlet Nextword prediction")
text=st.text_input("Enter the words ")
import numpy as np
import pickle
tokenizer=pickle.load(open('tokenizer.sav','rb'))
line=tokenizer.texts_to_sequences([text])
if len(line)>14:
    line=line[1:]
padding=np.array(pad_sequences(line,padding='pre',maxlen=14))
prediction=model.predict(padding) 
probability=np.argmax(prediction)
for word,index in tokenizer.word_index.items():
    if index==probability:
        st.write("The predicted word is",word)
        st.write(text,word)
        