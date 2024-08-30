
# Anime and Manga Recommendation System

This repository contains two projects: an Anime Recommendation System and a Manga Recommendation System. Both systems utilize Autoencoder models to suggest titles that are similar to a given input title based on various features such as genres, themes, type, score, members, and more.

## 1. Anime Recommendation System

### Overview
The Anime Recommendation System uses an Autoencoder model to generate recommendations for anime titles similar to a user-provided anime. The recommendations prioritize similarity, popularity (number of members), and the average score of the anime.

### Dataset
The dataset used for this project contains the following fields:
- `anime_id`: A unique identifier for each anime.
- `title`: The title of the anime.
- `type`: The format of the anime, such as TV, Movie, OVA, etc.
- `score`: The average score of the anime, typically from a large user base.
- `members`: The number of members who have added the anime to their list.
- `favorites`: The number of users who have marked the anime as a favorite.
- `genres`: The genres that the anime falls under, such as Action, Drama, Fantasy.
- `themes`: The themes associated with the anime, such as Supernatural, Mecha, School Life.
- `demographics`: The target demographic for the anime, such as Shounen, Seinen.
- `studios`: The animation studios responsible for producing the anime.

### Model
The model is an Autoencoder that reduces the dimensionality of the anime feature vectors, allowing for the extraction of latent features that are not immediately apparent. The compressed vectors are used to calculate the similarity between different anime titles using Cosine Similarity. The similarity score is then combined with the popularity (members) and score to prioritize recommendations.

### Usage
1. Run the script `AnimeRecomandation.py`.
2. Enter the title of an anime that you have watched.
3. The system will output a list of similar anime, ranked by similarity, popularity, and score.

### Dependencies
- `pandas`: For data manipulation and analysis.
- `scikit-learn`: For feature extraction and similarity calculations.
- `tensorflow`: For building and training the Autoencoder model.
- `scipy`: For handling sparse matrices.

### How to Run
```bash
python AnimeRecomandation.py
```

### License
This project is licensed under the MIT License.

## 2. Manga Recommendation System

### Overview
The Manga Recommendation System is built similarly to the Anime Recommendation System, using an Autoencoder model to suggest manga titles that are similar to a user-provided manga. The recommendations are also prioritized based on similarity, popularity (number of members), and average score.

### Dataset
The dataset used for this project contains the following fields:
- `manga_id`: A unique identifier for each manga.
- `title`: The title of the manga.
- `type`: The format of the manga, such as Manga, Light Novel, etc.
- `score`: The average score of the manga, typically from a large user base.
- `members`: The number of members who have added the manga to their list.
- `favorites`: The number of users who have marked the manga as a favorite.
- `genres`: The genres that the manga falls under, such as Action, Drama, Fantasy.
- `themes`: The themes associated with the manga, such as Supernatural, Mecha, School Life.
- `demographics`: The target demographic for the manga, such as Shounen, Seinen.
- `authors`: The authors who created the manga.

### Model
An Autoencoder model is used to reduce the dimensionality of the manga feature vectors, similar to the Anime Recommendation System. The model learns latent features that are useful for making recommendations. These features are used to compute similarities between manga titles using Cosine Similarity. Recommendations are then prioritized based on similarity, popularity (members), and score.

### Usage
1. Run the script `MangaRecomandation.py`.
2. Enter the title of a manga that you have read.
3. The system will output a list of similar manga, ranked by similarity, popularity, and score.

### Dependencies
- `pandas`: For data manipulation and analysis.
- `scikit-learn`: For feature extraction and similarity calculations.
- `tensorflow`: For building and training the Autoencoder model.
- `scipy`: For handling sparse matrices.

### How to Run
```bash
python MangaRecomandation.py
```

### License
This project is licensed under the MIT License.

## Conclusion
Both the Anime and Manga Recommendation Systems in this repository are designed to help users discover titles that they might enjoy based on their past preferences. By leveraging Autoencoder models, these systems are able to capture complex relationships between different titles and provide meaningful recommendations. Whether you are a fan of anime or manga, these tools can enhance your viewing or reading experience by suggesting content that is closely aligned with your tastes.

