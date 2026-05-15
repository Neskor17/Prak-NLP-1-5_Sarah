import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

# stopword bahasa indonesia
stop_words = set(stopwords.words('indonesian'))

# stemmer
factory = StemmerFactory()
stemmer = factory.create_stemmer()

def preprocess(text):
    # 1. case folding
    text = text.lower()

    # 2. cleaning
    text = re.sub(r'[^a-zA-Z\s]', '', text)

    # 3. tokenization
    tokens = word_tokenize(text)

    # 4. stopword removal
    tokens = [word for word in tokens if word not in stop_words]

    # 5. stemming
    text = " ".join(tokens)
    text = stemmer.stem(text)

    return text