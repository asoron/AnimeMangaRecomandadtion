import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import StandardScaler
import scipy.sparse as sp

import tensorflow as tf
from tensorflow.keras import layers, models

from sklearn.metrics.pairwise import cosine_similarity

# Veri yükleme ve ön işlemler
df = pd.read_csv('DataBase/anime_data.csv')

# Verileri sıralama ve eksik puanları çıkarma
df = df.sort_values(by='members', ascending=False)
df = df.dropna(subset=['score'])

# Eksik kategorik verileri doldurma
df[['type','genres', 'themes', 'demographics','studios']] = df[['type','genres', 'themes', 'demographics','studios']].fillna('Unknown')

# Metin verilerini vektörize etme
aniemGenres = [' '.join(genres) if isinstance(genres, list) else genres for genres in df['genres']]
animeThemes = [' '.join(themes) if isinstance(themes, list) else themes for themes in df['themes']]

tfidfVectorizer = TfidfVectorizer(stop_words=None)
countVectorizer = CountVectorizer(stop_words=None)

genres = tfidfVectorizer.fit_transform(aniemGenres)
themes = tfidfVectorizer.fit_transform(animeThemes)
studios = countVectorizer.fit_transform(df['studios'])
types = countVectorizer.fit_transform(df['type'])
demographics = countVectorizer.fit_transform(df['demographics'])

# Sayısal verileri ölçekleme
scaler = StandardScaler()  # Veya MinMaxScaler()
numeric_features = ['score', 'members', 'favorites']
df[numeric_features] = scaler.fit_transform(df[numeric_features])

# Metin vektörlerini birleştirme
text_features = sp.hstack([genres, themes, studios, types, demographics])

# Vektör ve sayısal özellikleri birleştirme
X = sp.hstack([text_features, df[numeric_features].values])

X = X.toarray()


input_dim = X.shape[1]  # Giriş boyutu (özellik sayısı)
encoding_dim = 128  # Sıkıştırılmış (latent) boyut

input_layer = layers.Input(shape=(input_dim,))
encoded = layers.Dense(encoding_dim, activation='relu')(input_layer)
decoded = layers.Dense(input_dim, activation='sigmoid')(encoded)

autoencoder = models.Model(input_layer, decoded)

# Encoder modelini tanımlama (özellik çıkarımı için)
encoder = models.Model(input_layer, encoded)

# Modeli derleme
autoencoder.compile(optimizer='adam', loss='mse')

# Modeli eğitme
history = autoencoder.fit(X, X, 
                          epochs=50, 
                          batch_size=256, 
                          shuffle=True, 
                          validation_split=0.2)

# Düşük boyutlu vektörleri çıkarma (latent space)
anime_vectors = encoder.predict(X)

# Çıkarılan vektörleri inceleme
print(anime_vectors.shape)  # (num_anime, encoding_dim)



# Girilen animeye en yakın animeleri bulma fonksiyonu
def find_similar_animes(anime_index, anime_vectors, top_n=10):
    cosine_similarities = cosine_similarity([anime_vectors[anime_index]], anime_vectors).flatten()
    similar_indices = cosine_similarities.argsort()[-top_n-1:-1][::-1]
    return similar_indices

# Girilen anime adını dataframe'de bulma fonksiyonu
def get_anime_index_by_name(anime_name, df):
    results = df[df['title'].str.contains(anime_name, case=False, na=False)]
    if not results.empty:
        return results.index[0], results.iloc[0]['title']
    else:
        return None, None

# Girilen animeye en yakın animeleri bulma fonksiyonu (Popülerlik ve puanı daha fazla dikkate alarak)
def find_similar_animes(anime_index, anime_vectors, df, top_n=10, similarity_weight=0.5, members_weight=0.3, score_weight=0.2):
    cosine_similarities = cosine_similarity([anime_vectors[anime_index]], anime_vectors).flatten()
    
    # Cosine similarity, members ve score'ı birleştirme
    df['similarity'] = cosine_similarities
    
    # Ağırlıklı ortalama hesaplama
    df['weighted_score'] = (similarity_weight * df['similarity']) + \
                           (members_weight * df['members']) + \
                           (score_weight * df['score'])
    
    similar_animes = df.iloc[df.index != anime_index]  # Girdiği animeyi dışarıda tut
    
    # Ağırlıklı puana göre sıralama
    similar_animes = similar_animes.sort_values(by='weighted_score', ascending=False)
    
    return similar_animes.head(top_n).index

# Benzer animeleri bulma ve gösterme fonksiyonu (popülerlik ve puan ön planda)
def show_similar_animes(anime_name, df, anime_vectors, top_n=5):
    anime_index, found_name = get_anime_index_by_name(anime_name, df)
    
    if anime_index is not None:
        print(f"\nBenzer animeler '{found_name}' için öneriler:")
        similar_animes = find_similar_animes(anime_index, anime_vectors, df, top_n=top_n)
        
        for i, index in enumerate(similar_animes):
            anime_title = df.iloc[index]['title']
            anime_score = df.iloc[index]['score']
            anime_members = df.iloc[index]['members']
            print(f"{i+1}. {anime_title} (Puan: {anime_score}, Üye Sayısı: {anime_members})")
    else:
        print("Anime bulunamadı. Lütfen tekrar deneyin.")

while True:
    anime_name = input("İzlediğiniz animenin ismini girin (Çıkmak için 'q' yazın): ")
    if anime_name.lower() == 'q':
        print("Program sonlandırıldı.")
        break
    show_similar_animes(anime_name, df, anime_vectors)
