# Finnish Forum discussion exploratory language analysis demo

#### Built using a BERT finnish sentiment model by [Nisan Coşkun](https://huggingface.co/nisancoskun/bert-finnish-sentiment-analysis-v2)

#### by Aapo Mikkola, 2025.04.02

#### What it does

1. Extract text from PDF (cached to a .txt). Uses PyPDF2.PdfReader.

2. Preprocess (Finnish stopwords + forum stopwords).

3. LDA topics with gensim.models.LdaMulticore (parallel LDA).

4. Embeddings via Sentence-Transformers paraphrase-multilingual-MiniLM-L12-v2 (multilingual).

5. Clustering & 2D views (KMeans, PCA/t-SNE).

6. Sentiment with Hugging Face pipeline("sentiment-analysis") (with Finnish/multilingual fallbacks).

7. Visualizations: topic word clouds (PNG), topic distributions/heatmap, cluster plots. Word clouds use the wordcloud library.

#### Configure

Edit the Config dataclass at the bottom of 10) PIPELINE:

- pdf_path / cache_path

- num_topics (LDA), kmeans_clusters

- embed_model (default: paraphrase-multilingual-MiniLM-L12-v2)

- figures_dir (output directory)
