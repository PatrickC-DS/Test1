import streamlit as st
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Embedding, Dense, GlobalAveragePooling1D
from tensorflow.keras.models import load_model
import pandas as pd

@st.cache_data
def load_modelCBOW() :
    model = load_model("117-04-Streamlit-word2vec.h5") 
    vectors = model.layers[0].trainable_weights[0].numpy()
    return vectors

@st.cache_data
def load_dict() :

    # Charger le DataFrame depuis le fichier CSV
    df = pd.read_csv("117-04-Streamlit-word2idx.csv")
    # Créer le dictionnaire word2idx
    word2idx = dict(zip(df['word'], df['index']))
    # Créer idx2word directement à partir du DataFrame
    idx2word = dict(zip(df['index'], df['word']))
    vocab_size = df.shape[0]
    
    return df, vocab_size, word2idx, idx2word

import numpy as np
from sklearn.preprocessing import Normalizer

def dot_product(vec1, vec2):
    return np.sum((vec1*vec2))

def cosine_similarity(vec1, vec2):
    return dot_product(vec1, vec2)/np.sqrt(dot_product(vec1, vec1)*dot_product(vec2, vec2))

def find_closest(word_index, vectors, number_closest):
    list1=[]
    query_vector = vectors[word_index]
    for index, vector in enumerate(vectors):
        if not np.array_equal(vector, query_vector):
            dist = cosine_similarity(vector, query_vector)
            list1.append([dist,index])
    return np.asarray(sorted(list1,reverse=True)[:number_closest])

def compare(index_word1, index_word2, index_word3, vectors, number_closest):
    list1=[]
    query_vector = vectors[index_word1] - vectors[index_word2] + vectors[index_word3]
    normalizer = Normalizer()
    query_vector =  normalizer.fit_transform([query_vector], 'l2')
    query_vector= query_vector[0]
    for index, vector in enumerate(vectors):
        if not np.array_equal(vector, query_vector):
            dist = cosine_similarity(vector, query_vector)
            list1.append([dist,index])
    return np.asarray(sorted(list1,reverse=True)[:number_closest])

def print_closest(word, number=10):
    index_closest_words = find_closest(word2idx[word], vectors, number)
    for index_word in index_closest_words :
        st.write(idx2word[index_word[1]]," -- ",index_word[0])

df, vocab_size, word2idx, idx2word = load_dict()    # depuis cache 
vectors = load_modelCBOW()                          # depuis cache 
print("vocab_size=", vocab_size)
print("Nb word=", df['word'].nunique())
print("Nb idx=", df['index'].nunique())


st.title("Modèle Word2Vec")

st.sidebar.title("Sommaire")
pages=["Dictionnaire", "Zombie", "Autre Mots"]
page=st.sidebar.radio("Pages", pages)

if page == pages[0] : 
    st.write("### Dictionnaire")
    st.dataframe(df.head(10))
    st.write("Shape")
    st.write(df.shape)
    st.write("Describe")
    st.dataframe(df.describe())
    if st.checkbox("Afficher les NA") :
        st.dataframe(df.isna().sum())

if page == pages[1] : 
    st.write("### Mots proche de zombie")
    print_closest('zombie')

if page == pages[2] : 
    choix = list(df['word'].head(10000-1)) # on avait limiter le'entrainement à 10000 inputs max
    mot = st.selectbox('Choix du mot', choix)
    st.write('Le mot choisi est :', mot)
    print_closest(mot)

def comment() : # si je ne definis pas une fonction, le commentaire est considéré comme une variable afffichée par  !!!
    """
    #------------------------------------------------------------------------------
    # Exécution en local
    pip install streamlit  # une fois
    streamlit run 117-04-StreamlitPredict.py
    # => web browser avec http://localhost:8501/
    #------------------------------------------------------------------------------
   
     # Accès depuis le domaine public avec serveur sur Windows
    pip install pyngrok
    # Créer un compte ngrok sur le site officiel => colin35patrick@gmail.com
    # Copier le token disponible dans l'onglet "Your Authtoken"
    ngrok config add-authtoken 2x5jKEY2nTe4MvWlJv2zrgx38Mp_4pwMpZnY58RoKmxT1396q
    streamlit run 117-04-StreamlitPredict.py    # Dans un terminal Vscode
    npx localtunnel — port 8501                 # Dans un autre terminal Vscode
    # => ngrok offre une adresse publique avec Forwarding : https://6979-176-173-201-116.ngrok-free.app
    #------------------------------------------------------------------------------
    
    # Accès depuis le domaine public avec serveur sur colab
    # Connection à colab
    !pip install streamlit  # une fois
    !pip install pyngrok
    !ngrok config add-authtoken 2x5jKEY2nTe4MvWlJv2zrgx38Mp_4pwMpZnY58RoKmxT1396q
    # importer les fichiers csv et h5 (voir streamlit_app.py)
    from google.colab import files
    uploaded_model = files.upload() # 117-04-Streamlit-word2vec.h5
    uploaded_csv = files.upload()  # 117-04-Streamlit-word2idx.csv

    from pyngrok import ngrok 
    public_url = ngrok.connect(addr='8501') # et non port='8501'
    public_url

    # Création du fichier streamlit_app.py
    %%writefile streamlit_app.py 
    import streamlit as st    # et la suite du code copié 

    # lancement du serveur et du tunnel
    !streamlit run /content/streamlit_app.py & npx localtunnel --port 8501
    #    Local URL: http://localhost:8501
    #    Network URL: http://172.28.0.12:8501
    #    External URL: http://34.75.92.249:8501
    #    ⠴⠦your url is: https://afraid-days-ring.loca.lt
    #    Web browser avec l'adresse de "your url is ..." le mot de passe étant l'adresse ip external URL, ex 34.75.92.249
    #------------------------------------------------------------------------------
    
    # Accès depuis le domaine public avec serveur sur GitHub

    """