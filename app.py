import streamlit as st
import joblib
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Download required NLTK data (only needs to run once)
nltk.download('punkt')
nltk.download('stopwords')

# Load the trained model and vectorizer
try:
    model = joblib.load('sentiment_model.pkl')
    vectorizer = joblib.load('vectorizer.pkl')
except FileNotFoundError:
    st.error("Error: 'sentiment_model.pkl' or 'vectorizer.pkl' not found. Please ensure these files are in the same directory.")
    st.stop()

# Preprocess the text data
def preprocess_text(text):
    text = text.lower()
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token.isalpha() and token not in stop_words]
    return ' '.join(tokens)

# Function to predict sentiment
def predict_sentiment(review):
    preprocessed = preprocess_text(review)
    features = vectorizer.transform([preprocessed])
    prediction = model.predict(features)[0]
    return 'Positive' if prediction == 1 else 'Negative'

# Streamlit interface
st.set_page_config(
    page_title="Movie Review Sentiment Analysis",
    page_icon=":clapper:",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for enhanced styling
st.markdown(
    """
    <style>
    .reportview-container {
        background: linear-gradient(135deg, #E0FFFF, #ADD8E6);
    }
    .big-font {
        font-size:2.5rem !important;
        font-weight: bold;
        color: #004080;
    }
    .stTextArea textarea {
        background-color: #f0f8ff;
        color: #262730;
        border: 1px solid #4682B4;
    }
    .stButton>button {
        color: #ffffff;
        background-color: #558B2F;
        border: none;
        border-radius: 6px;
        padding: 10px 20px;
    }
    .stButton>button:hover {
        background-color: #33691E;
        color: #ffffff;
    }
    .stSuccess {
        color: #155724;
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        padding: 15px;
        border-radius: 5px;
    }
    .stWarning {
        color: #856404;
        background-color: #fff3cd;
        border: 1px solid #ffeeba;
        padding: 15px;
        border-radius: 5px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Title and Introduction
st.markdown("<p class='big-font'>Movie Review Sentiment Analysis</p>", unsafe_allow_html=True)
st.write("Enter a movie review below to discover whether it's generally positive or negative.")

# Sidebar for additional information or settings
with st.sidebar:
    st.header("About This App")
    st.image("image to review.jpg", use_container_width=True)
    st.write("This app uses a machine learning model to analyze the sentiment of movie reviews.")
    st.write("It predicts whether the sentiment of a review is positive or negative based on the text input.")
    st.markdown("[Learn more about Sentiment Analysis](https://en.wikipedia.org/wiki/Sentiment_analysis)")

# Text area for user input
review = st.text_area("Enter your Movie Review here:", height=200, placeholder="Write your review here...")

# Button to trigger prediction
if st.button("Predict Sentiment"):
    if review.strip() == "":
        st.warning("Please enter a review in the text area.")
    else:
        sentiment = predict_sentiment(review)
        st.success(f"Predicted Sentiment: **{sentiment}** :movie_camera:")

# Footer
st.markdown("---")
st.write("Created BY M.Haseeb using Streamlit")
