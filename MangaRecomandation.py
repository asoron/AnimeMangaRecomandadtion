import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import StandardScaler
import scipy.sparse as sp
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.metrics.pairwise import cosine_similarity

# Veri yükleme ve ön işlemler
df = pd.read_csv('DataBase/manga_data.csv')

# Verileri sıralama ve eksik puanları çıkarma
df = df.sort_values(by='members', ascending=False)
df = df.dropna(subset=['score'])

# Eksik kategorik verileri doldurma
df[['type','genres', 'themes', 'demographics', 'authors']] = df[['type','genres', 'themes', 'demographics', 'authors']].fillna('Unknown')

# Metin verilerini vektörize etme
mangaGenres = [' '.join(genres) if isinstance(genres, list) else genres for genres in df['genres']]
mangaThemes = [' '.join(themes) if isinstance(themes, list) else themes for themes in df['themes']]

tfidfVectorizer = TfidfVectorizer(stop_words=None)
countVectorizer = CountVectorizer(stop_words=None)

genres = tfidfVectorizer.fit_transform(mangaGenres)
themes = tfidfVectorizer.fit_transform(mangaThemes)
authors = countVectorizer.fit_transform(df['authors'])
types = countVectorizer.fit_transform(df['type'])
demographics = countVectorizer.fit_transform(df['demographics'])

# Sayısal verileri ölçekleme
scaler = StandardScaler()  # Veya MinMaxScaler()
numeric_features = ['score', 'members', 'favorites']
df[numeric_features] = scaler.fit_transform(df[numeric_features])

# Metin vektörlerini birleştirme
text_features = sp.hstack([genres, themes, authors, types, demographics])

# Vektör ve sayısal özellikleri birleştirme
X = sp.hstack([text_features, df[numeric_features].values])

X = X.toarray()

# Autoencoder modelini tanımlama
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
manga_vectors = encoder.predict(X)

# Çıkarılan vektörleri inceleme
print(manga_vectors.shape)  # (num_manga, encoding_dim)

# Girilen manga'ya en yakın mangaları bulma fonksiyonu (Benzerlik, popülerlik ve puanı dikkate alarak)
def find_similar_mangas(manga_index, manga_vectors, df, top_n=10, similarity_weight=0.5, members_weight=0.3, score_weight=0.2):
    cosine_similarities = cosine_similarity([manga_vectors[manga_index]], manga_vectors).flatten()
    
    # Cosine similarity, members ve score'ı birleştirme
    df['similarity'] = cosine_similarities
    
    # Ağırlıklı ortalama hesaplama
    df['weighted_score'] = (similarity_weight * df['similarity']) + \
                           (members_weight * df['members']) + \
                           (score_weight * df['score'])
    
    similar_mangas = df.iloc[df.index != manga_index]  # Girdiği mangayı dışarıda tut
    
    # Ağırlıklı puana göre sıralama
    similar_mangas = similar_mangas.sort_values(by='weighted_score', ascending=False)
    
    return similar_mangas.head(top_n).index

# Benzer mangaları bulma ve gösterme fonksiyonu (popülerlik ve puan ön planda)
def show_similar_mangas(manga_name, df, manga_vectors, top_n=5):
    manga_index, found_name = get_manga_index_by_name(manga_name, df)
    
    if manga_index is not None:
        print(f"\nBenzer mangalar '{found_name}' için öneriler:")
        similar_mangas = find_similar_mangas(manga_index, manga_vectors, df, top_n=top_n)
        
        for i, index in enumerate(similar_mangas):
            manga_title = df.iloc[index]['title']
            manga_score = df.iloc[index]['score']
            manga_members = df.iloc[index]['members']
            print(f"{i+1}. {manga_title} (Puan: {manga_score}, Üye Sayısı: {manga_members})")
    else:
        print("Manga bulunamadı. Lütfen tekrar deneyin.")

# Girilen manga adını dataframe'de bulma fonksiyonu
def get_manga_index_by_name(manga_name, df):
    results = df[df['title'].str.contains(manga_name, case=False, na=False)]
    if not results.empty:
        return results.index[0], results.iloc[0]['title']
    else:
        return None, None

# While döngüsü ile sürekli manga önerisi alma
while True:
    manga_name = input("İzlediğiniz manganın ismini girin (Çıkmak için 'q' yazın): ")
    if manga_name.lower() == 'q':
        print("Program sonlandırıldı.")
        break
    show_similar_mangas(manga_name, df, manga_vectors)
