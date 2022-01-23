import string
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk import word_tokenize
from lxml import etree
from nltk import WordNetLemmatizer
from nltk.corpus import stopwords

xml_path = "news.xml"

tree = etree.parse(xml_path)
root = tree.getroot()
corpus = root[0]
lemmatizer = WordNetLemmatizer()
bag_of_words = list()
headlines = list()
for news in corpus:
    for value in news:
        lem_text = list()
        final_text = list()
        final_text_pos = list()
        if value.get("name") == "head":
            headlines.append(value.text + ":")
        if value.get("name") == "text":
            tokenized_text = word_tokenize(value.text.lower())
            tokenized_text = sorted(tokenized_text, reverse=True)

            for token in tokenized_text:
                token = lemmatizer.lemmatize(token)
                lem_text.append(token)
            for _word in lem_text:
                if _word not in stopwords.words("english") and _word not in list(string.punctuation):
                    final_text.append(_word)
            for word in final_text:
                if nltk.pos_tag([word])[0][1] == "NN":
                    final_text_pos.append(word)

            #  old task solution
            #  tokenized_text_counter = Counter(final_text_pos)
            #  top_5 = tokenized_text_counter.most_common(5)
            #  for entry in top_5:
            #      print(entry[0], end=" ")
            #  print()

            document = " "
            document = document.join(final_text_pos)

            bag_of_words.append(document)

vectorizer = TfidfVectorizer(input='content', use_idf=True, lowercase=True, analyzer='word', ngram_range=(1, 1),
                             stop_words=None, vocabulary=None, min_df=0.01, max_df=0.60)
tfidf_matrix = vectorizer.fit_transform(bag_of_words)
matrix_array = tfidf_matrix.toarray()

terms = vectorizer.get_feature_names()

df = pd.DataFrame(data=matrix_array, index=headlines, columns=terms)
df = df.T


counter = 0
for headline in headlines:
    result = " "

    headline_df = df = pd.DataFrame(data=[matrix_array[counter], terms], index=["TFIDF Value", "Term"])
    headline_df = headline_df.T
    headline_df.sort_values(by=["TFIDF Value", "Term"], ascending=False, inplace=True)
    top_5_df_new = headline_df.head(5)
    top_5_df_new = top_5_df_new["Term"]
    top_5_df_new = top_5_df_new.values
    print(headline)
    print(" ".join(top_5_df_new))

    counter = counter + 1
