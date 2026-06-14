import pickle

movies = pickle.load(
    open("models/movies.pkl", "rb")
)

similarity = pickle.load(
    open("models/similarity.pkl", "rb")
)

svd_model = pickle.load(
    open("models/svd_model.pkl", "rb")
)


def hybrid_recommend(movie_name, user_id=1):

    matched = movies[
        movies['title'].str.lower()
        == movie_name.lower()
    ]

    if len(matched) == 0:
        return []

    movie_index = matched.index[0]

    distances = similarity[movie_index]

    movie_list = sorted(
        list(enumerate(distances)),
        reverse=True,
        key=lambda x: x[1]
    )[1:21]

    recommendations = []

    for movie_idx, content_score in movie_list:

        movie_title = movies.iloc[movie_idx].title

        # Simple hybrid score
        hybrid_score = content_score

        recommendations.append(
            (
                movie_title,
                hybrid_score
            )
        )

    recommendations = sorted(
        recommendations,
        key=lambda x: x[1],
        reverse=True
    )

    return recommendations[:5]

