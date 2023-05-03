# Calculate inertia score for each topic
import matplotlib.pyplot as plt
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer()

X = vectorizer.fit_transform(sentences_df['PROCESSED_SENTENCE'])
inertia_scores = []
models = []
n_components_range = range(min_topics, max_topics+step_size, step_size)

for n_components in n_components_range:
    print(f"n_components={n_components}")
    models.append(LatentDirichletAllocation(n_components=n_components, max_iter=5,
                                    learning_method='online',
                                    learning_offset=50.,
                                    random_state=0))
    models[n_components-2].fit(X)
    inertia_score = models[n_components-2].score(X)
    inertia_scores.append(inertia_score)

plt.plot(n_components_range, inertia_scores)
plt.xlabel('Number of topics')
plt.ylabel('Inertia score')
plt.show()
