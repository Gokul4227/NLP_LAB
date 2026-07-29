import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import treebank
from nltk.tag import hmm
from nltk.tag import UnigramTagger, DefaultTagger

# Download resources
nltk.download('punkt')
nltk.download('treebank')

# -------------------------
# Train HMM POS Tagger
# -------------------------
print("Training HMM model...")

# Load training data
train_data = treebank.tagged_sents()

# Train HMM
trainer = hmm.HiddenMarkovModelTrainer()
hmm_tagger = trainer.train_supervised(train_data)

# Create backoff tagger
default_tagger = DefaultTagger("NN")
unigram_tagger = UnigramTagger(train_data, backoff=default_tagger)

print("Training Completed!\n")

# -------------------------
# User Input (Multi-line Paragraph)
# -------------------------
print("Enter a paragraph (Paste multiple lines)")
print("Type END on a new line when finished:\n")

lines = []

while True:
    line = input()

    if line.strip().upper() == "END":
        break

    lines.append(line)

# Preserve sentence boundaries
paragraph = "\n".join(lines)

# Split into sentences
sentences = sent_tokenize(paragraph)

print("\nPOS Tagged Output:\n")

total_words = 0

for sentence in sentences:

    tokens = word_tokenize(sentence)

    total_words += len(tokens)

    # First tag using HMM
    hmm_result = hmm_tagger.tag(tokens)

    tagged = []

    for word, tag in hmm_result:

        # If HMM predicts NNP too aggressively
        if tag == "NNP":

            backup = unigram_tagger.tag([word])[0]

            if backup[1] is not None:
                tag = backup[1]

        tagged.append((word, tag))

    print("\nSentence:")
    print(sentence)

    print("\nTags:")

    for word, tag in tagged:
        print(f"{word:<20} -> {tag}")

    print("\n" + "-" * 50)

print("\nTotal Words:", total_words)

# -------------------------
# Tag Meanings
# -------------------------
print("\nCommon Tag Meanings:")
print("NN  -> Noun")
print("NNS -> Plural Noun")
print("NNP -> Proper Noun")
print("VB  -> Verb")
print("VBG -> Verb (-ing)")
print("JJ  -> Adjective")
print("RB  -> Adverb")
print("DT  -> Determiner")
print("PRP -> Pronoun")