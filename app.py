# import streamlit as st
# import pickle
# import string
# from nltk.corpus import stopwords
# import nltk
# from nltk.stem.porter import PorterStemmer
# ps = PorterStemmer()
# def transform_text(text):
#     text = text.lower()
#     text = nltk.word_tokenize(text)
#     y = []
#     for i in text:
#         if i.isalnum():
#             y.append(i)
#     text = y[:]
#     y.clear()
#     for i in text:
#         if i not in stopwords.words('english') and i not in string.punctuation:
#             y.append(i)
#     text = y[:]
#     y.clear()
#     for i in text:
#         y.append(ps.stem(i))
#     return " ".join(y)
# tfidf = pickle.load(open('vectorizer (1).pkl' , 'rb'))
# model = pickle.load(open('model (1).pkl' , 'rb'))
#
# st.title("spam-message-or-not")
# in_sms = st.text_input("Enter your message")
#
# # 1. preprocess
#
# transform_sms = transform_text(in_sms)
#
# # 2. vectorize
# vector_input = tfidf.transform([transform_sms])
#
# # 3. predict
# result = model.predict(vector_input)[0]
#
# # 4. Display
# if result == 1:
#     st.header("Spam message")
# else:
#     st.header("Not Spam")


import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

nltk.download('punkt')
nltk.download('stopwords')

ps = PorterStemmer()

# Page settings
st.set_page_config(page_title="Spam Detector", page_icon="📩", layout="centered")

st.markdown("""
<style>

/* Main background */
.stApp {
    background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
    background-attachment: fixed;
}

/* Title styling */
h1 {
    text-align: center;
    color: white;
    font-size: 50px;
}

/* Subtitle */
p {
    text-align: center;
    color: #d1d1d1;
    font-size: 18px;
}

/* Input box */
textarea {
    border-radius: 12px !important;
    border: 2px solid #4CAF50 !important;
    padding: 10px !important;
}

/* Button style */
.stButton>button {
    width: 100%;
    height: 50px;
    font-size: 18px;
    border-radius: 12px;
    background: linear-gradient(90deg,#00c6ff,#0072ff);
    color: white;
    border: none;
    transition: 0.3s;
}

/* Button hover */
.stButton>button:hover {
    background: linear-gradient(90deg,#ff512f,#dd2476);
    transform: scale(1.05);
}

/* Result box spacing */
.result-box {
    padding:20px;
    border-radius:10px;
    margin-top:20px;
}

</style>
""", unsafe_allow_html=True)

# Title
st.markdown("<h1 style='text-align: center;'>📩 Spam Message Detector</h1>", unsafe_allow_html=True)
st.write("Check whether a message is **Spam or Not Spam** instantly.")

# Text preprocessing
def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

# Load model
tfidf = pickle.load(open('vectorizer (1).pkl','rb'))
model = pickle.load(open('model (1).pkl','rb'))

# Input box
input_sms = st.text_area("✉ Enter your message here")

if st.button("🔍 Predict"):

    transformed_sms = transform_text(input_sms)

    vector_input = tfidf.transform([transformed_sms])

    result = model.predict(vector_input)[0]

    st.write("---")

    if result == 1:
        st.error("🚨 Spam Message Detected")
    else:

        st.success("✅ This is Not Spam")
