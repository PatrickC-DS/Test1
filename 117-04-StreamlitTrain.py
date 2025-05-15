import pandas as pd
import re
import unicodedata
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


df = pd.read_csv("MovieReview.csv")
print(df.head())
print(df.shape)

# Supression colonne cible
df = df.drop('sentiment', axis=1)

nltk.download('punkt')
stop_words = stopwords.words('english')

# Converts the unicode file to ascii
def unicode_to_ascii(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn')

def preprocess_sentence(w):
    w = unicode_to_ascii(w.lower().strip())
    # Insertion d'un espacee après chaque signe de ponctuation
    # eg: "he is a boy." => "he is a boy ."
    w = re.sub(r"([?.!,¿])", r" \1 ", w)
    # réduction des espaces multiple à 1
    w = re.sub(r'[" "]+', " ", w)
    # replacing everything with space except (a-z, A-Z, ".", "?", "!", ",")
    w = re.sub(r"[^a-zA-Z?.!]+", " ", w)
    w = re.sub(r'\b\w{0,2}\b', '', w)

    # Effacement des stopword
    mots = word_tokenize(w.strip())
    mots = [mot for mot in mots if mot not in stop_words]
    return ' '.join(mots).strip()

# Formatage des tokens et supression des mots sans contenu
df.review = df.review.apply(lambda x :preprocess_sentence(x))
print(df.head())

# Vectorisation du texte (texte en une séquence d'entiers, chaque entier étant l'index d'un token dans un dictionnaire)
import tensorflow as tf
tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=10000) # max 10000 input pour l'entrainement, pas le max du dictionnaire, ni du df sauvé 
tokenizer.fit_on_texts(df.review)                       

word2idx = tokenizer.word_index     # word -> index
idx2word = tokenizer.index_word     # index -> word
vocab_size = tokenizer.num_words    # taille (pas 10000 mais 72540)

# Enregistrement du dictionnaire
dict_name = "117-04-Streamlit-word2idx.csv"
word2idx_df = pd.DataFrame(list(word2idx.items()), columns=['word', 'index'])
# Sauvegarder le DataFrame en fichier CSV
word2idx_df.to_csv(dict_name, index=False)  # 7240 lignes

"""
Implémentation d'un modèle Word2Vec particulier : le modèle Continuous Bag Of Words (CBOW). 
Le modèle CBOW cherche à prédire un mot grâce à son contexte, c'est-à-dire grâce aux mots proches de lui dans le texte. 
Les inputs du modèle sont les mots du contexte et l'output du modèle est une probabilité de prédiction du mot cible.
Le modèle CBOW est un réseau de neurones à 3 couches : une couche d'input, une couche cachée et une couche ouput. 
La couche cachée est constituée d'une couche Embedding qui transforme chaque mot input en un vecteur d'embedding, 
de tel sorte que la matrice d'embedding est apprise au fur et à mesure de l'entrainement. 
Il y a aussi une couche de Pooling (GlobalAveragePooling1D) qui somme les différents embeddings pour obtenir un résultat de bonne dimension. 
Enfin la prédiction du mot cible est faite grâce à une couche Dense.
"""
import numpy as np

def sentenceToData(tokens, WINDOW_SIZE):
    window = np.concatenate((np.arange(-WINDOW_SIZE,0),np.arange(1,WINDOW_SIZE+1)))
    X,Y=([],[])
    for word_index, word in enumerate(tokens) :
        if ((word_index - WINDOW_SIZE >= 0) and (word_index + WINDOW_SIZE <= len(tokens) - 1)) :
            X.append(word)
            Y.append([tokens[word_index-i] for i in window])
    return X, Y

WINDOW_SIZE = 5  # la taille de la fenêtre de contexte (nombre de mots autour du mot cible).

X, Y = ([], [])
for review in df.review:
    for sentence in review.split("."):
        word_list = tokenizer.texts_to_sequences([sentence])[0] # word_list = list des tokens de la phrase
        #print("W=", word_list)
        if len(word_list) >= WINDOW_SIZE:                       # Si au moins 5 mots
            Y1, X1 = sentenceToData(word_list, WINDOW_SIZE//2)  # Y1 les mots au milieu, X mots (1 liste / mot de Y1) qui entourent les mots de Y1
            X.extend(X1)
            Y.extend(Y1)
    
X = np.array(X).astype(int)
y = np.array(Y).astype(int).reshape([-1,1])

print("X=", X)
print("y=", y)

# Architecture du modèle
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Embedding, Dense, GlobalAveragePooling1D

embedding_dim = 300
model = Sequential()
model.add(Embedding(vocab_size, embedding_dim))
model.add(GlobalAveragePooling1D())
model.add(Dense(vocab_size, activation='softmax'))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(X, y, batch_size = 128, epochs=5)

# Enregistrement du modèle
model_name = "117-04-Streamlit-word2vec.h5"
model.save(model_name)

print(f"Modèle sauvé sous {model_name} avec {dict_name}")