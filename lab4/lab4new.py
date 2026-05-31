import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors


# 1. ЗАВАНТАЖЕННЯ ДАНИХ
def load_data(path):
    return np.loadtxt(path)


# 2. CENTNN 
class CentNN:
    def __init__(self, n_clusters):
        self.k = n_clusters
        self.centroids = None

    def fit(self, X, max_epochs=100):
        np.random.seed(42)
        self.centroids = X[np.random.choice(len(X), self.k, replace=False)]

        for epoch in range(max_epochs):
            clusters = [[] for _ in range(self.k)]

            for x in X:
                distances = np.linalg.norm(x - self.centroids, axis=1)
                winner = np.argmin(distances)
                clusters[winner].append(x)

            new_centroids = []

            for i, cluster in enumerate(clusters):
                if len(cluster) > 0:
                    new_centroids.append(np.mean(cluster, axis=0))
                else:
                    new_centroids.append(self.centroids[i])

            new_centroids = np.array(new_centroids)

            if np.allclose(self.centroids, new_centroids):
                print(f"CentNN: зупинка на епосі {epoch}")
                break

            self.centroids = new_centroids

        return self

    def predict(self, X):
        labels = []
        for x in X:
            distances = np.linalg.norm(x - self.centroids, axis=1)
            labels.append(np.argmin(distances))
        return np.array(labels)


# 3. ВІЗУАЛІЗАЦІЯ
def plot_clusters(X, labels, centroids, title):
    plt.figure()
    plt.scatter(X[:, 0], X[:, 1], c=labels, s=10)
    plt.scatter(centroids[:, 0], centroids[:, 1], marker='x')
    plt.title(title)
    plt.show()


# 4. КЛАСТЕРИЗАЦІЯ
def run_clustering():
    datasets = {
        "unbalance": "unbalance2.txt", # Дані з різною щільністю
        "s2": "s2.txt",   # Синтетичні дані
        "a2": "a2.txt"    # Дані з високою щільністю та близькими кластерами
    } 

    for name, path in datasets.items():
        print(f"\nDATASET: {name}")
        try:
            X = load_data(path)
        except FileNotFoundError:
            print(f"Файл {path} не знайдено, пропускаємо кластеризацію.")
            continue

        # CentNN
        centnn = CentNN(n_clusters=6)
        centnn.fit(X)
        labels_centnn = centnn.predict(X)

        print("CentNN clusters:", np.bincount(labels_centnn))
        plot_clusters(X, labels_centnn, centnn.centroids, f"CentNN - {name}")

        # KMeans
        kmeans = KMeans(n_clusters=6, random_state=42)
        labels_kmeans = kmeans.fit_predict(X)

        print("KMeans clusters:", np.bincount(labels_kmeans))
        plot_clusters(X, labels_kmeans, kmeans.cluster_centers_, f"KMeans - {name}")


# 5. СИСТЕМА РЕКОМЕНДАЦІЙ
def load_books_data():
    books = pd.read_csv(
        "BX-Books.csv",
        encoding="latin-1",
        sep=";",
        on_bad_lines="skip",
        low_memory=False
    )

    users = pd.read_csv(
        "BX-Users.csv",
        encoding="latin-1",
        sep=";",
        on_bad_lines="skip"
    )

    ratings = pd.read_csv(
        "BX-Book-Ratings.csv",
        encoding="latin-1",
        sep=";",
        on_bad_lines="skip"
    )

    # очищення
    books = books[['ISBN', 'Book-Title', 'Book-Author', 'Year-Of-Publication', 'Publisher']]
    books.columns = ['isbn', 'title', 'author', 'year', 'publisher']

    users = users[['User-ID', 'Location', 'Age']]
    users.columns = ['user_id', 'location', 'age']

    ratings = ratings[['User-ID', 'ISBN', 'Book-Rating']]
    ratings.columns = ['user_id', 'isbn', 'rating']

    return books, users, ratings


# 6. ФІЛЬТРАЦІЯ + PIVOT
def create_pivot(ratings, min_user=200, min_book=50):
    user_counts = ratings['user_id'].value_counts()
    active_users = user_counts[user_counts > min_user].index

    book_counts = ratings['isbn'].value_counts()
    popular_books = book_counts[book_counts > min_book].index

    filtered = ratings[
        (ratings['user_id'].isin(active_users)) &
        (ratings['isbn'].isin(popular_books))
    ]

    pivot = filtered.pivot_table(index='isbn', columns='user_id', values='rating').fillna(0)
    return pivot


# 7. KNN РЕКОМЕНДАЦІЇ
def build_knn(pivot):
    model = NearestNeighbors(metric='euclidean')
    model.fit(pivot)
    return model


def recommend(book_id, pivot, model, n=10):
    distances, indices = model.kneighbors([pivot.loc[book_id]], n_neighbors=n+1)
    recs = []
    for i in indices.flatten()[1:]:
        recs.append(pivot.index[i])
    return recs


# --- НОВИЙ ДОДОТКОВИЙ ФУНКЦІОНАЛ ДЛЯ ДЕКОДУВАННЯ КНИГ ---
def get_book_info(isbn, books_df):
    """Повертає назву та автора книги за її ISBN. Якщо немає в базі — повертає сам ISBN."""
    book_row = books_df[books_df['isbn'] == isbn]
    if not book_row.empty:
        title = book_row.iloc[0]['title']
        author = book_row.iloc[0]['author']
        return f"'{title}' - {author} ({isbn})"
    return f"ISBN: {isbn} (Назва відсутня в BX-Books)"


# 8. ЗАПУСК РЕКОМЕНДАЦІЙ (ОНОВЛЕНО)
def run_recommendation():
    books, users, ratings = load_books_data()
    pivot = create_pivot(ratings, 200, 50)

    print("Pivot shape:", pivot.shape)

    model = build_knn(pivot)
    sample_books = pivot.index[:5]  # Зменшив до 5 для компактності логів

    print("\n--- РЕЗУЛЬТАТИ РЕКОМЕНДАЦІЙ ---")
    for book_isbn in sample_books:
        book_desc = get_book_info(book_isbn, books)
        print(f"\nКнига-запит: {book_desc}")
        
        recs = recommend(book_isbn, pivot, model)
        print("Рекомендовані книги:")
        for idx, rec_isbn in enumerate(recs, 1):
            print(f"  {idx}. {get_book_info(rec_isbn, books)}")


# 9. ЕКСПЕРИМЕНТИ (ОНОВЛЕНО)
def run_experiments():
    books, users, ratings = load_books_data()

    thresholds_users = [100, 150] # Скорочено списки, щоб консоль не потонула в спамі
    thresholds_books = [25, 50]

    print("\n--- ЗАПУСК ЕКСПЕРИМЕНТІВ ---")
    for u in thresholds_users:
        for b in thresholds_books:
            print(f"\n>>> Налаштування фільтрації: Користувачі > {u} відгуків, Книги > {b} відгуків")

            pivot = create_pivot(ratings, u, b)

            if pivot.shape[0] < 3:
                print("Занадто мало даних для побудови рекомендацій.")
                continue

            model = build_knn(pivot)
            sample = pivot.index[:2] # Беремо 2 книги для демонстрації

            for book_isbn in sample:
                target_desc = get_book_info(book_isbn, books)
                recs = recommend(book_isbn, pivot, model)
                
                print(f"  Книга: {target_desc}")
                print("  Топ-3 схожих:")
                for rec_isbn in recs[:3]:
                    print(f"    -> {get_book_info(rec_isbn, books)}")


if __name__ == "__main__":
    print("1. КЛАСТЕРИЗАЦІЯ")
    run_clustering()

    print("\n2. РЕКОМЕНДАЦІЇ")
    run_recommendation()

    print("\n3. ЕКСПЕРИМЕНТИ")
    run_experiments()