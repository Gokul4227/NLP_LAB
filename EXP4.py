import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD

# Read documents
n = int(input("Enter number of documents: "))
docs = []

for i in range(n):
    doc = input(f"Enter document {i+1}: ")
    docs.append(doc)

# Read query
query = input("\nEnter search query: ")

# TF-IDF Vectorization
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(docs)

# Transform query
query_vec = vectorizer.transform([query])

# Classical IR: TF-IDF + Cosine Similarity
tfidf_scores = cosine_similarity(query_vec, X)

print("\n----- TF-IDF Similarity Scores -----")
for i, score in enumerate(tfidf_scores[0]):
    print(f"Document {i+1}: {score:.3f}")

# LSA (Latent Semantic Analysis)
if X.shape[1] > 1:
    n_components = min(2, X.shape[1] - 1)
else:
    n_components = 1

svd = TruncatedSVD(n_components=n_components, random_state=42)
X_lsa = svd.fit_transform(X)
query_lsa = svd.transform(query_vec)

# Non-Classical IR: LSA + Cosine Similarity
lsa_scores = cosine_similarity(query_lsa, X_lsa)

print("\n----- LSA Similarity Scores -----")
for i, score in enumerate(lsa_scores[0]):
    print(f"Document {i+1}: {score:.3f}")

# Most relevant document
best_doc = np.argmax(lsa_scores[0])

print("\n----- Most Relevant Document -----")
print(f"Document {best_doc + 1}:")
print(docs[best_doc])