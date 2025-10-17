from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

KNOWLEDGE_GRAPH = {
    "what is the capital of france?": "The capital of France is Paris.",
    "what is the tallest mountain in the world?": "Mount Everest is the tallest mountain in the world.",
    "who wrote 'hamlet'?": "William Shakespeare wrote 'Hamlet'.",
    "what is the currency of japan?": "The currency of Japan is the Japanese Yen."
}

def search_knowledge_graph(query: str):
    """
    Searches the knowledge graph for the most relevant answer to a query.
    """
    documents = list(KNOWLEDGE_GRAPH.keys())
    documents.append(query)

    vectorizer = TfidfVectorizer().fit_transform(documents)
    vectors = vectorizer.toarray()

    query_vector = vectors[-1]
    document_vectors = vectors[:-1]

    similarities = cosine_similarity([query_vector], document_vectors)
    most_similar_index = similarities.argmax()

    if similarities[0][most_similar_index] > 0.1:  # Threshold to avoid irrelevant answers
        best_match_question = documents[most_similar_index]
        return KNOWLEDGE_GRAPH[best_match_question]
    else:
        return "I'm sorry, I don't have an answer to that question."
