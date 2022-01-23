from sklearn.feature_extraction.text import TfidfVectorizer

with open("data/dataset/input.txt", "r") as file:

    vectorizer = TfidfVectorizer(input='file', use_idf=True, lowercase=True,
                                 analyzer='word', ngram_range=(1, 1),
                                 stop_words=None)

    tfidf_matrix = vectorizer.fit_transform([file])

    print(tfidf_matrix[(0, 10)])
