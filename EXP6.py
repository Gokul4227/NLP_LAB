import nltk
from nltk.tokenize import word_tokenize
from sklearn.metrics import precision_score, recall_score, f1_score

# Download required NLTK resources
nltk.download("punkt")
nltk.download("punkt_tab")

# Biomedical relation keywords
RELATION_KEYWORDS = {"treats", "reduces", "controls", "helps"}

# Get user input
sentence = input("Enter a biomedical sentence: ").strip()
actual_relation = int(input("Enter Actual Relation (1 = Yes, 0 = No): "))

# Tokenize the sentence
tokens = word_tokenize(sentence.lower())

# Display tokens
print("\nTokenized Words:")
print(tokens)

# Predict relation
predicted_relation = 1 if any(word in RELATION_KEYWORDS for word in tokens) else 0

# Display prediction
print(f"\nPredicted Relation: {predicted_relation}")

# Evaluate prediction
y_true = [actual_relation]
y_pred = [predicted_relation]

precision = precision_score(y_true, y_pred, zero_division=0)
recall = recall_score(y_true, y_pred, zero_division=0)
f1 = f1_score(y_true, y_pred, zero_division=0)

# Display evaluation metrics
print("\nEvaluation Metrics")
print("-------------------")
print(f"Precision : {precision:.2f}")
print(f"Recall    : {recall:.2f}")
print(f"F1-Score  : {f1:.2f}")