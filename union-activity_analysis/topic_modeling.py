import pandas as pd
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from bertopic.representation import KeyBERTInspired
from umap import UMAP
from hdbscan import HDBSCAN

posts_data_topredict = pd.read_csv("../data/posts_data_for_emotion_prediction.csv")

subset_data = posts_data_topredict.sample(n=5000, random_state=42)

sentence_model = SentenceTransformer("all-distilroberta-v1") # this truncates long text automatically, in our case we have less than 1% of data which is too long.
representation_model = KeyBERTInspired()
umap = UMAP(random_state=0) 
hdbscan_model = HDBSCAN(min_cluster_size=50, metric='euclidean', cluster_selection_method='eom', prediction_data=True)
topic_model = BERTopic(embedding_model=sentence_model, representation_model=representation_model, umap_model=umap, hdbscan_model=hdbscan_model)

docs = subset_data["text"].tolist()

# Encode in batches
embeddings = sentence_model.encode(docs, batch_size=64, show_progress_bar=True)

topics, probs = topic_model.fit_transform(docs, embeddings=embeddings)

topic_model.save("./model/", serialization="safetensors", save_ctfidf=True, save_embedding_model=sentence_model)

info_topics = topic_model.get_topic_info()
# save info
info_topics.to_csv("../data/topic_info_sample.csv", index=False)

hierarchical_topics = topic_model.hierarchical_topics(docs)
# save hierarchical topics
hierarchical_topics.to_csv("../data/hierarchical_topics_sample.csv", index=False)