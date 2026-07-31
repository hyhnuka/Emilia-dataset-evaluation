01: BEGIN
02:     // 1. DATA LOADING & PREPROCESSING
03:     LOAD "dataset-QA_7500.jsonl" INTO df
04:     INITIALIZE Sastrawi Stemmer & StopwordRemover
05:     DEFINE custom_stopwords (pronomina, kata tanya, verba umum)
06:     
07:     FUNCTION preprocess_text(text):
08:         text <- Lowercase(text)
09:         text <- Remove_Non_Alphabetic_Characters(text)
10:         text <- Sastrawi_Stemming(text)
11:         tokens <- Tokenize(text)
12:         tokens <- Filter(tokens NOT IN all_stopwords AND length > 3)
13:         RETURN Join(tokens)
14:     END FUNCTION
15: 
16:     df['preprocessed'] <- preprocess_text(df['question'] + " " + df['answer'])
17:     REMOVE empty documents FROM df
18: 
19:     // 2. INITIAL MODEL TRAINING
20:     embedding_model <- Load model
21:     embeddings <- embedding_model.encode(df['preprocessed'])
22:     
23:     umap_model <- UMAP(neighbors=11, components=6, metric='cosine')
24:     hdbscan_model <- HDBSCAN(min_cluster_size=28, min_samples=7, prediction_data=True)
25:     topic_model <- BERTopic(umap_model, hdbscan_model, vectorizer_model)
26:     
27:     topics, probs <- topic_model.fit_transform(df['preprocessed'], embeddings)
28: 
29:     // 3. DEFINE EVALUATION FUNCTIONS
30:     FUNCTION evaluate_metrics(df_input):
31:         docs_tokenized <- [doc.split() FOR doc IN df_input WHERE Topic != -1]
32:         topic_words <- Extract_Top_10_Words(df_input)
33:         
34:         diversity <- Count_Unique(topic_words) / Count_Total(topic_words)
35:         coherence_cv <- compute_gensim_coherence(topic_words, docs_tokenized)
36:         
37:         RETURN {coh: coherence_cv, div: diversity, n_out: Count(Topic == -1)}
38:     END FUNCTION
39: 
40:     metrics_initial <- evaluate_metrics(df) // Baseline Evaluation
41: 
42:     // 4. OUTLIER HANDLING PREPARATION
43:     FUNCTION compute_centroids(df, embeddings):
44:         FOR EACH topic_id IN valid_topics:
45:             idx_samples <- Random_Select(topic_indices, max_docs=100)
46:             centroids[topic_id] <- Mean(embeddings[idx_samples])
47:         END FOR
48:         RETURN centroids
49:     END FUNCTION
50:     topic_centroids <- compute_centroids(df, embeddings)
51: 
52:     // 5. THRESHOLD OPTIMIZATION (SIMULATION)
53:     candidates <- [0.60, 0.65, 0.70, 0.75]
54:     FOR EACH thr IN candidates:
55:         df_sim <- Simulate_Reassignment(df_outliers, topic_centroids, thr)
56:         res <- evaluate_metrics(df_sim)
57:         
58:         assign_score <- count_reassigned(df_sim) / total_outliers
59:         total_score <- (0.5 * assign_score) + (0.3 * res.coh) + (0.2 * res.div)
60:         STORE total_score, thr INTO results_list
61:     END FOR
62:     optimal_thr <- thr WITH MAX(total_score)
63: 
64:     // 6. TWO-STAGE REASSIGNMENT EXECUTION
65:     FOR EACH outlier IN df_outliers:
66:         max_sim, best_topic <- Cosine_Similarity(outlier, topic_centroids)
67:         IF max_sim >= optimal_thr:
68:             Assign Topic = best_topic, Stage = "High Confidence"
69:         ELSE IF max_sim >= (optimal_thr - 0.10):
70:             Assign Topic = best_topic, Stage = "Medium Confidence"
71:         END IF
72:     END FOR
73: 
74:     // 7. SELECTIVE WORD RECALCULATION
75:     FOR EACH topic_id IN affected_topics:
76:         impact_ratio <- new_docs_count / total_docs_in_topic
77:         IF impact_ratio >= 0.15:
78:             Recalculate_Keywords_TFIDF(topic_id, top_n=10)
79:         ELSE:
80:             Keep_Original_Keywords(topic_id)
81:         END IF
82:     END FOR
83: 
84:     // 8. FINAL EVALUATION & OUTPUT
85:     metrics_final <- evaluate_metrics(df_updated)
86:     DISPLAY metrics_initial VS metrics_final
87:     SAVE df_updated TO CSV
88: END



from umap import UMAP
from hdbscan import HDBSCAN
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

from bertopic import BERTopic
from bertopic.representation import KeyBERTInspired
from bertopic.vectorizers import ClassTfidfTransformer

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import AgglomerativeClustering, KMeans
from scipy.cluster.hierarchy import linkage, dendrogram

from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

import pandas as pd
import numpy as np
import json
import re
from collections import defaultdict, Counter

from gensim.corpora import Dictionary
from gensim.models import CoherenceModel

import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')

print("Libraries loaded successfully")

data_rows = []
with open("dataset-QA_7500.jsonl", 'r', encoding='utf-8') as f:
    for line in f:
        data_rows.append(json.loads(line))

df = pd.DataFrame(data_rows)

print(f"Dataset loaded: {len(df):,} documents")
print(f"Columns: {df.columns.tolist()}")

stemmer_factory = StemmerFactory()
stemmer = stemmer_factory.create_stemmer()

stopword_factory = StopWordRemoverFactory()
stopwords_indonesia = stopword_factory.get_stop_words()

additional_stopwords = [
    # pronomina
    'aku','saya','kamu','dia','mereka','kita','kalian','orang','orang-orang','sama-sama',

    # kata_tanya_umum
    'cara','gimana','bagaimana','mana','apa','kenapa','ngapain','ngapa','yang','sih','tanya',
    'saja','dong','gitu','kayak','aja','cuma','kayaknya','begitu', 'kalau', 'sama', 'jadi', 

    # kata_verba_umum
    'coba','bikin','buat','biar','lihat','lihat-lihat','ajar','lihatnya','punya','buatnya',
    'buatkan','cari','temu','ketemu','bilang','ucap','cerita','ceritain','tulis','ganti',
    'ambil','tarik','berasa', 'memang', 'kira', 'alih', 'nuna', 'ikut', 'gambar', 'suara', 'nyanyi'
]
stopwords_indonesia = set(stopwords_indonesia).union(set(additional_stopwords))

def preprocess_text(text):
    if pd.isna(text):
        return ""
    text = text.lower()
    text = re.sub(r'[^a-z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    text = stemmer.stem(text)
    words = text.split()
    words = [w for w in words if w not in stopwords_indonesia and len(w) > 3]
    return ' '.join(words)

print("Preprocessing documents...")
df['document_preprocessed'] = (df['question'] + " " + df['answer']).apply(preprocess_text)
df = df[df['document_preprocessed'].str.len() > 0].reset_index(drop=True)
print(f"Valid documents: {len(df):,}")

### try 2
print("Loading SentenceTransformer model...")
# embedding_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
embedding_model = SentenceTransformer("paraphrase-multilingual-mpnet-base-v2")

print("Generating embeddings...")
documents = df['document_preprocessed'].tolist()
embeddings = embedding_model.encode(documents, show_progress_bar=True)
print(f"Embeddings: {embeddings.shape}")

umap_model = UMAP(n_neighbors=11, n_components=6, min_dist=0.0, metric='cosine', random_state=42)
hdbscan_model = HDBSCAN(min_cluster_size=28, min_samples=7, metric='euclidean', cluster_selection_method='eom', prediction_data=True)
vectorizer_model = CountVectorizer(stop_words=list(stopwords_indonesia), min_df=2, ngram_range=(1, 2))

topic_model = BERTopic(
    embedding_model=embedding_model,
    umap_model=umap_model,
    hdbscan_model=hdbscan_model,
    vectorizer_model=vectorizer_model,
    top_n_words=10,
    nr_topics="auto",
    calculate_probabilities=True,
    verbose=True
)

print("="*70)
print("Training BERTopic")
print("="*70)
topics, probs = topic_model.fit_transform(documents, embeddings)
print("Training complete")

topic_info = topic_model.get_topic_info()
n_topics = len(topic_info[topic_info['Topic'] != -1])
outlier_count = topic_info[topic_info['Topic'] == -1]['Count'].values[0] if -1 in topic_info['Topic'].values else 0

print(f"\nInitial Results:")
print(f"Topics: {n_topics}")
print(f"Outliers: {outlier_count:,} ({outlier_count/len(df)*100:.2f}%)")

// 7. Evaluation Functions
def topic_words_from_dataframe(df_input, top_n=10):
    """Extract topic keywords from DataFrame with error handling"""
    df_valid = df_input[df_input["Topic"] != -1].copy()
    topic_words = {}

    for topic_id in sorted(df_valid["Topic"].unique()):
        try:
            top_words_str = df_valid[df_valid["Topic"] == topic_id]["Top_n_words"].iloc[0]

            # Handle different formats and empty values
            if pd.isna(top_words_str) or top_words_str in ["", "outlier", "no_words"]:
                continue

            words = [w.strip() for w in str(top_words_str).split(" - ") if w.strip() and w.strip() not in ["outlier", "no_words", ""]]

            if words:  # Only add if we have valid words
                topic_words[topic_id] = words[:top_n]
        except Exception as e:
            continue

    return topic_words

def compute_topic_diversity(topic_words_dict):
    """Calculate topic diversity (unique words / total words)"""
    if not topic_words_dict:
        return 0.0
    all_words = [w for words in topic_words_dict.values() for w in words]
    if not all_words:
        return 0.0
    return len(set(all_words)) / len(all_words)

def compute_coherence_cv(docs_tokenized, topic_words_dict):
    """Calculate coherence C_v using Gensim with error handling"""
    if not topic_words_dict or not docs_tokenized:
        return 0.0

    # Filter out topics with empty/invalid words
    valid_topics = []
    for topic_id, words in topic_words_dict.items():
        if words and all(isinstance(w, str) and w.strip() for w in words):
            valid_topics.append(words)

    if not valid_topics:
        return 0.0

    try:
        dictionary = Dictionary(docs_tokenized)
        corpus = [dictionary.doc2bow(tokens) for tokens in docs_tokenized]

        if not corpus:
            return 0.0

        cm = CoherenceModel(
            topics=valid_topics,
            texts=docs_tokenized,
            dictionary=dictionary,
            corpus=corpus,
            coherence="c_v"
        )
        return cm.get_coherence()
    except Exception as e:
        print(f"  [Warning] Coherence calculation failed: {str(e)[:50]}")
        return 0.0

def evaluate_metrics(df_input, stage_name=""):
    """Comprehensive evaluation of topic modeling metrics"""
    # Only use valid documents (not outliers)
    df_valid_docs = df_input[df_input["Topic"] != -1].copy()

    if len(df_valid_docs) == 0:
        sep_line = "=" * 70
        print(f"\n{sep_line}")
        print(f"Evaluation: {stage_name}")
        print(sep_line)
        print("ERROR: No valid documents")
        print(sep_line)
        return {"stage": stage_name, "coherence": 0, "diversity": 0, "n_topics": 0, "n_outliers": len(df_input), "outlier_ratio": 100}

    docs_tokenized = [doc.split() for doc in df_valid_docs["Document"].tolist()]
    topic_words = topic_words_from_dataframe(df_input, top_n=10)

    if not topic_words:
        diversity = 0.0
        coherence_cv = 0.0
    else:
        diversity = compute_topic_diversity(topic_words)
        coherence_cv = compute_coherence_cv(docs_tokenized, topic_words)

    n_outliers = len(df_input[df_input["Topic"] == -1])
    outlier_ratio = n_outliers / len(df_input) * 100
    n_topics = len(topic_words)

    sep_line = "=" * 70
    print(f"\n{sep_line}")
    print(f"Evaluation: {stage_name}")
    print(sep_line)
    print(f"Coherence C_v : {coherence_cv:.4f}")
    print(f"Diversity     : {diversity:.4f}")
    print(f"Topics        : {n_topics}")
    print(f"Outliers      : {n_outliers:,} ({outlier_ratio:.2f}%)")
    print(sep_line)

    return {"stage": stage_name, "coherence": coherence_cv, "diversity": diversity, "n_topics": n_topics, "n_outliers": n_outliers, "outlier_ratio": outlier_ratio}

print("Evaluation functions defined")
metrics_initial = evaluate_metrics(df, "Initial Results")

//8. Compute Topic Centroid
def compute_topic_centroids(df_input, embeddings_array, max_docs=100):
    df_valid = df_input[df_input['Topic'] != -1].copy()
    topic_centroids = {}
    for topic_id in sorted(df_valid['Topic'].unique()):
        topic_indices = df_valid[df_valid['Topic'] == topic_id].index.tolist()
        if len(topic_indices) > max_docs:
            topic_indices = np.random.choice(topic_indices, max_docs, replace=False).tolist()
        topic_embeddings = embeddings_array[topic_indices]
        centroid = np.mean(topic_embeddings, axis=0)
        topic_centroids[topic_id] = centroid
    print(f"Computed centroids for {len(topic_centroids)} topics")
    return topic_centroids

topic_centroids = compute_topic_centroids(df, embeddings, max_docs=100)

//9. Outlier Reassignemnt
def reassign_outliers_twostage(df_outliers, outlier_embeddings, topic_centroids, threshold_high=0.70, threshold_medium=0.60):
    """
    Reassign outliers to nearest topics using two-stage threshold
    """
    print(f"\n{'='*70}")
    print(f"Outlier Reassignment")
    print(f"{'='*70}")
    print(f"High confidence threshold  : {threshold_high}")
    print(f"Medium confidence threshold: {threshold_medium}")
    print(f"Total outliers             : {len(df_outliers):,}")

    reassignments = []
    stage1_count = 0
    stage2_count = 0

    for idx, outlier_emb in enumerate(outlier_embeddings):
        max_sim = -1
        best_topic = -1

        for topic_id, centroid in topic_centroids.items():
            sim = cosine_similarity([outlier_emb], [centroid])[0][0]
            if sim > max_sim:
                max_sim = sim
                best_topic = topic_id

        # Stage 1: High confidence 
        if max_sim >= threshold_high:
            new_topic = best_topic
            stage1_count += 1
            reassigned = True
        # Stage 2: Medium confidence
        elif max_sim >= threshold_medium:
            new_topic = best_topic
            stage2_count += 1
            reassigned = True
        else:
            new_topic = -1
            reassigned = False

        reassignments.append({
            'index': df_outliers.iloc[idx].name,
            'question': df_outliers.iloc[idx]['question'],
            'document': df_outliers.iloc[idx]['Document'],
            'old_topic': -1,
            'new_topic': new_topic,
            'similarity': max_sim,
            'reassigned': reassigned,
            'stage': 'high' if max_sim >= threshold_high else ('medium' if reassigned else 'rejected')
        })

    df_reassigned = pd.DataFrame(reassignments)
    total_reassigned = df_reassigned['reassigned'].sum()
    still_outlier = len(df_reassigned) - total_reassigned

    print(f"\n{'='*70}")
    print(f"Reassignment Results")
    print(f"{'='*70}")
    print(f"Stage 1 (High confidence)   : {stage1_count:,} ({stage1_count/len(df_reassigned)*100:.1f}%)")
    print(f"Stage 2 (Medium confidence) : {stage2_count:,} ({stage2_count/len(df_reassigned)*100:.1f}%)")
    print(f"Total reassigned            : {total_reassigned:,} ({total_reassigned/len(df_reassigned)*100:.1f}%)")
    print(f"Still outliers              : {still_outlier:,} ({still_outlier/len(df_reassigned)*100:.1f}%)")
    print(f"Average similarity          : {df_reassigned['similarity'].mean():.4f}")
    print(f"{'='*70}")

    return df_reassigned

print("Two-stage reassignment function defined")

//10. Threshold selection
#for find_optimal_threshold_real
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np

def get_topic_words(df, n_top_words=10):
    """
    Recalculate topic words after reassignment.
    Returns dict: topic_id -> list of top-N words
    """
    topic_words = {}
    vectorizer = CountVectorizer()
    docs = df['Document'].tolist()
    X = vectorizer.fit_transform(docs)
    vocab = np.array(vectorizer.get_feature_names_out())

    for topic_id in sorted(df['Topic'].unique()):
        if topic_id == -1:
            continue
        topic_docs = df[df['Topic'] == topic_id].index.tolist()
        if not topic_docs:
            topic_words[topic_id] = []
            continue

        topic_matrix = X[topic_docs].sum(axis=0)  
        top_idx = np.asarray(topic_matrix).ravel().argsort()[::-1][:n_top_words]
        topic_words[topic_id] = vocab[top_idx].tolist()

    return topic_words

# Execute reassignment with optimal threshold
df_reassigned = reassign_outliers_twostage(
    df_outlier,
    outlier_embeddings,
    topic_centroids,
    threshold_high=optimal_threshold,
    threshold_medium=max(0.60, optimal_threshold - 0.10)
)

11. Update dataset
df_updated = df.copy()
topics_with_new_docs = set()

for idx, row in df_reassigned[df_reassigned['reassigned']].iterrows():
    df_updated.loc[row['index'], 'Topic'] = row['new_topic']
    topics_with_new_docs.add(row['new_topic'])

print(f"Topics that received outliers: {len(topics_with_new_docs)}")
print(f"Topics preserved: {metrics_initial['n_topics'] - len(topics_with_new_docs)}")

metrics_after_reassign = evaluate_metrics(df_updated, "After Outlier Reassignment")

//12. Representative word calculation
def recalculate_representative_words_minimal(df_input, df_original, topics_to_recalc, min_new_docs_ratio=0.15, top_n=10):
    """
    Selectively recalculate representative words for impacted topics only
    """
    print(f"\n{'='*70}")
    print(f"Selective Recalculation")
    print(f"{'='*70}")
    print(f"Minimum new docs ratio for recalc: {min_new_docs_ratio*100:.0f}%")

    df_valid = df_input[df_input['Topic'] != -1].copy()
    vectorizer = CountVectorizer(stop_words=list(stopwords_indonesia), max_features=1000)

    new_top_words = {}
    actually_recalculated = 0

    for topic_id in sorted(df_valid['Topic'].unique()):
        # Check if this topic received new docs
        if topic_id not in topics_to_recalc:
            old_words = df_original[df_original['Topic'] == topic_id]['Top_n_words'].iloc[0]
            new_top_words[topic_id] = [w.strip() for w in old_words.split(' - ')][:top_n]
            continue

        # Check impact ratio: how many new docs vs total docs in topic
        current_size = len(df_valid[df_valid['Topic'] == topic_id])
        original_size = len(df_original[df_original['Topic'] == topic_id])
        new_docs_count = current_size - original_size

        if new_docs_count <= 0:
            impact_ratio = 0
        else:
            impact_ratio = new_docs_count / current_size

        # Only recalculate if impact is significant
        if impact_ratio < min_new_docs_ratio:
            old_words = df_original[df_original['Topic'] == topic_id]['Top_n_words'].iloc[0]
            new_top_words[topic_id] = [w.strip() for w in old_words.split(' - ')][:top_n]
            print(f"  Topic {topic_id}: Impact {impact_ratio*100:.1f}% < threshold, KEEPING original")
            continue

        # Recalculate for high-impact topics
        topic_docs = df_valid[df_valid['Topic'] == topic_id]['Document'].tolist()

        if len(topic_docs) < 3:
            old_words = df_original[df_original['Topic'] == topic_id]['Top_n_words'].iloc[0]
            new_top_words[topic_id] = [w.strip() for w in old_words.split(' - ')][:top_n]
            continue

        try:
            doc_term_matrix = vectorizer.fit_transform(topic_docs)
            tfidf_transformer = TfidfTransformer()
            tfidf_matrix = tfidf_transformer.fit_transform(doc_term_matrix)
            avg_tfidf = np.asarray(tfidf_matrix.mean(axis=0)).flatten()
            feature_names = vectorizer.get_feature_names_out()
            top_indices = avg_tfidf.argsort()[-top_n:][::-1]
            top_words = [feature_names[i] for i in top_indices]

            new_top_words[topic_id] = top_words
            actually_recalculated += 1
            print(f"  Topic {topic_id}: Impact {impact_ratio*100:.1f}%, RECALCULATED ({len(topic_docs)} docs, +{new_docs_count} new)")

        except Exception as e:
            old_words = df_original[df_original['Topic'] == topic_id]['Top_n_words'].iloc[0]
            new_top_words[topic_id] = [w.strip() for w in old_words.split(' - ')][:top_n]

    print(f"\n{'='*70}")
    print(f"Recalculation Summary")
    print(f"{'='*70}")
    print(f"Topics that received outliers    : {len(topics_to_recalc)}")
    print(f"Actually recalculated (high impact): {actually_recalculated}")
    print(f"Kept original (low impact)        : {len(topics_to_recalc) - actually_recalculated}")
    print(f"Total topics with original words  : {len(new_top_words) - actually_recalculated}")
    print(f"Preservation rate                 : {(len(new_top_words) - actually_recalculated) / len(new_top_words) * 100:.1f}%")
    print(f"{'='*70}")

    return new_top_words

# Recalculate with minimal approach
new_representative_words = recalculate_representative_words_minimal(
    df_updated,
    df,
    topics_with_new_docs,
    min_new_docs_ratio=0.15,
    top_n=10
)
# Update DataFrame
for topic_id, words in new_representative_words.items():
    new_words_str = " - ".join(words)
    df_updated.loc[df_updated['Topic'] == topic_id, 'Top_n_words'] = new_words_str

metrics_after_recalc = evaluate_metrics(df_updated, "Final Results")

