import nltk
import string
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import wordnet
from nltk import pos_tag

# ----------------------------
# Download resources if needed
# ----------------------------
resources = [
    ("tokenizers/punkt", "punkt"),
    ("corpora/wordnet", "wordnet"),
    ("taggers/averaged_perceptron_tagger_eng", "averaged_perceptron_tagger_eng")
]

for path, resource in resources:
    try:
        nltk.data.find(path)
    except LookupError:
        nltk.download(resource)


# ----------------------------
# Convert POS tags
# ----------------------------
def get_wordnet_pos(tag):
    if tag.startswith("J"):
        return wordnet.ADJ
    elif tag.startswith("V"):
        return wordnet.VERB
    elif tag.startswith("N"):
        return wordnet.NOUN
    elif tag.startswith("R"):
        return wordnet.ADV
    return wordnet.NOUN


# ----------------------------
# Text processing function
# ----------------------------
def process_text(text):

    if not text.strip():
        print("Error: Empty input")
        return

    # Tokenization
    tokens = word_tokenize(text)

    # Remove punctuation
    clean_tokens = [
        word for word in tokens
        if word not in string.punctuation
    ]

    # Stemming
    stemmer = PorterStemmer()
    stemmed = [
        stemmer.stem(word)
        for word in clean_tokens
    ]

    # POS-aware Lemmatization
    lemmatizer = WordNetLemmatizer()

    tagged_words = pos_tag(clean_tokens)

    lemmatized = [
        lemmatizer.lemmatize(
            word,
            get_wordnet_pos(tag)
        )
        for word, tag in tagged_words
    ]

    # Output
    print("\n" + "=" * 50)

    print("\nOriginal Text:")
    print(text)

    print("\nTokens:")
    print(tokens)

    print("\nClean Tokens:")
    print(clean_tokens)

    print("\nStemmed Words:")
    print(stemmed)

    print("\nLemmatized Words:")
    print(lemmatized)

    print("\nWord Comparison:")

    for original, stem, lemma in zip(
        clean_tokens,
        stemmed,
        lemmatized
    ):
        print(
            f"{original:<15} "
            f"Stem → {stem:<12} "
            f"Lemma → {lemma}"
        )

    print("\n" + "=" * 50)


# ----------------------------
# Main
# ----------------------------
if __name__ == "__main__":

    sentence = input(
        "Enter a sentence: "
    )

    process_text(sentence)