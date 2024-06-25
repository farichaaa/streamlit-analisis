import streamlit as st
from textblob import TextBlob
import re
import nltk
import openpyxl
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report

# Unduh korpus yang diperlukan untuk TextBlob
nltk.download('punkt')

# Fungsi untuk menganalisis sentimen dengan TextBlob
def analyze_sentiment(text):
    blob = TextBlob(text)
    if blob.sentiment.polarity > 0:
        return "Positif"
    elif blob.sentiment.polarity < 0:
        return "Negatif"
    else:
        return "Netral"

# Fungsi untuk melabeli komentar dengan aturan berbasis kata kunci
def label_comment(comment):
    comment = comment.lower()
    sentiment_dict = {
        'kecewa': -0.4, 'rugi': -1, 'buruk': -10, 'jelek': -4, 'lelet': -0.7,
        'gagal': -4.7, 'parah': -0.6, 'mahal': -0.3, 'tolong': -0.1, 'hilang': -0.3,
        'gajelas': -0.3, 'gj': -0.3, 'promo': 0.6, 'kadang': -0.1, 'maling': -0.5,
        'ganggu': 0.3, 'sedot': -0.5, 'bagus': 0.5, 'pulsa': 0, 'potong': -1,
        'baik': 0.5, 'kntl': -1, 'ngelag': -0.8, 'salah': -0.5, 'bintang': 0,
        'benerin': -0.4, 'lambat': -0.8, 'siput': -0.4, 'mati': -5, 'minimal': -0.3,
        'susah': -0.6, 'nagih': -0.6, 'capek': -0.7, 'kacau': -0.3, 'tagih': -0.3,
        'mantap': 1, 'puas': 0.9, 'sampah': -0.5, 'sulit': -0.6, 'aneh': -0.4,
        'tidak': -5, 'jeglek': -4, 'permasalahan': -1, 'diputus': -2.5, 'diganti': -2.3,
        'pembatalan': -3.9, 'dicabut': -3.2, 'pemadam': -2, 'gangguan': -0.9,
        'batal': -2, 'tagihan': -2.6, 'periksa': -0.8, 'rusak': -5
    }
    intensifiers = {'sangat': 2, 'sekali': 2, 'banget': 2, 'amat': 2, 'terlalu': 1.5}
    negations = ['tidak', 'bukan', 'kurang', 'belum']
    words = re.findall(r'\w+', comment)
    sentiment_score = 0
    negated = False
    intensity_multiplier = 1

    for word in words:
        if word in negations:
            negated = True
            continue
        if word in intensifiers:
            intensity_multiplier = intensifiers[word]
            continue
        if word in sentiment_dict:
            score = sentiment_dict[word]
            if negated:
                score = -score
            score *= intensity_multiplier
            sentiment_score += score
        negated = False
        intensity_multiplier = 1

    if sentiment_score > 0.5:
        return 'positif'
    elif sentiment_score < -0.5:
        return 'negatif'
    else:
        return 'netral'

# Fungsi untuk menampilkan halaman login
def show_login_page():
    st.title("Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        if username in st.session_state["users"] and st.session_state["users"][username] == password:
            st.session_state["logged_in"] = True
            st.experimental_rerun()
        else:
            st.error("Username atau password salah")

# Fungsi untuk menampilkan halaman registrasi
def show_register_page():
    st.title("Registrasi")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    confirm_password = st.text_input("Konfirmasi Password", type="password")
    if st.button("Registrasi"):
        if password == confirm_password:
            if username in st.session_state["users"]:
                st.error("Username sudah ada. Silakan pilih username lain.")
            else:
                st.session_state["users"][username] = password
                st.success("Registrasi berhasil! Silakan login.")
                st.session_state["page"] = "login"
                st.experimental_rerun()
        else:
            st.error("Password tidak cocok")

# Fungsi untuk menampilkan halaman utama analisis sentimen
def show_sentiment_analysis_page():
    st.title("Analisis Sentimen Komentar")
    user_input = st.text_area("Masukkan komentar Anda di sini:")
    if st.button("Cek Sentimen"):
        if user_input:
            textblob_result = analyze_sentiment(user_input)
            rule_based_result = label_comment(user_input)
            ml_result = classify_comment(user_input)
            st.write("Sentimen TextBlob:", textblob_result)
            st.write("Sentimen Rule-Based:", rule_based_result)
            st.write("Sentimen Machine Learning:", ml_result)

    st.write("Atau unggah file untuk analisis sentimen:")
    uploaded_file = st.file_uploader("Pilih file", type=["xlsx"])
    if uploaded_file is not None:
        data = load_data(uploaded_file)
        st.write("File berhasil diunggah. Hasil analisis sentimen:")
        for comment in data:
            st.write(comment)
            st.write("Sentimen TextBlob:", analyze_sentiment(comment))
            st.write("Sentimen Rule-Based:", label_comment(comment))
            st.write("Sentimen Machine Learning:", classify_comment(comment))
            st.write("------")

    # Menambahkan tombol logout di halaman analisis sentimen
    if st.button("Logout"):
        st.session_state["logged_in"] = False
        st.experimental_rerun()

# Fungsi utama untuk mengatur navigasi halaman
def main():
    if "page" not in st.session_state:
        st.session_state["page"] = "main"
    if "logged_in" not in st.session_state:
        st.session_state["logged_in"] = False
    if "users" not in st.session_state:
        st.session_state["users"] = {}

    st.sidebar.title("Navigation")
    if st.session_state["logged_in"]:
        show_sentiment_analysis_page()
        if st.sidebar.button("Logout"):
            st.session_state["logged_in"] = False
            st.experimental_rerun()
    else:
        page = st.sidebar.radio("Go to", ["Login", "Register", "Home"])
        if page == "Login":
            show_login_page()
        elif page == "Register":
            show_register_page()
        else:
            st.title("Welcome to Sentiment Analysis App")
            st.write("Please select an option from the sidebar to continue.")

# Memuat dataset dan melatih model pembelajaran mesin
def load_data(file):
    df = pd.read_excel(file)
    comments = df.iloc[:, 0].tolist()  # Assuming comments are in the first column
    return comments

data = load_data('komenfix.xlsx')
comments_list = data
labels_list = [label_comment(comment) for comment in comments_list]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(comments_list, labels_list, test_size=0.2, random_state=42)

# Vectorize the text data
vectorizer = CountVectorizer()
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

# Train a Naive Bayes classifier
clf = MultinomialNB()
clf.fit(X_train_vectorized, y_train)
y_pred = clf.predict(X_test_vectorized)
print(classification_report(y_test, y_pred))

# Fungsi untuk mengklasifikasikan komentar baru dengan model pembelajaran mesin
def classify_comment(comment):
    vectorized_comment = vectorizer.transform([comment])
    prediction = clf.predict(vectorized_comment)
    return prediction[0]

if __name__ == "__main__":
    main()
