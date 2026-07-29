import nltk
from nltk import word_tokenize, pos_tag

# Download required resources
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger_eng')

# Input legal text
text = input("Enter legal text: ")

# Tokenize and POS tag
tokens = word_tokenize(text)
tags = pos_tag(tokens)

print("\nDetected Named Entities:")

count = 0

for word, tag in tags:
    if tag == "NNP":
        print(word, "-> ENTITY")
        count += 1

# User enters actual number of entities
actual = int(input("\nEnter actual number of entities: "))

# Calculate simple accuracy
accuracy = (min(count, actual) / max(count, actual)) * 100 if max(count, actual) != 0 else 100

print("\nPredicted Entities:", count)
print("NER Accuracy:", round(accuracy, 2), "%")