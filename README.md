# Fine Tuning Embedding Models for Retrieval on Domain Specific Data

<img src="https://miro.medium.com/v2/resize:fit:1400/0*AjX-xfa4UvNVu9js.jpg" width=600>

Embedding models are the backbone of modern Retrieval Augmented Generation pipelines, supplying a language model with the most similar and relevant context from a knowledgebase to aide it's generation. These are commonly used for querying over and finding insights among large quantities of unstructured data.

More often than not, we default to standard and generalized embedding models to convert our data into dense vector representations, which are then stored in a vector database and retrieved at runtime. And while these models are quite powerful to start, they suffer in performance when applied to domain specific or niche content- often failing to retrieve the most relevant or useful documents from an end user perspective. This error compounds as it is passed to a language model, which will confidently answer with erroneous data.

To address this, it's possible to fine tune open source embedding models on your own knowledgebase data to boost retrieved document, with minimal data prep using [Sentence Transformers](https://sbert.net/). In this notebook we'll walk through how I was able to boost my embedding model performance upwards of 60+% across standard information retrieval metrics for unseen queries through:

1. Preparing a synthetic dataset of positive question + chunk pairs
2. Manipulating and preparing the dataset for training and evaluators
3. Evaluating the base performance of the embedding model
4. Fine tuning the embedding model on our data with Matryoshka Representation Learning
5. Publishing the fine tuned model to Hugging Face
6. Evaluating the performance of our fine-tuned model

The resulting model has been published here: [AdamLucek/ModernBERT-embed-base-legal-MRL](https://huggingface.co/AdamLucek/ModernBERT-embed-base-legal-MRL)  
Along with the dataset: [AdamLucek/legal-rag-positives-synthetic](https://huggingface.co/datasets/AdamLucek/legal-rag-positives-synthetic)

This notebook is inspired by and pulls methodology and code snippets from Philipp Schmid's blog post: [*Fine-tune Embedding models for Retrieval Augmented Generation (RAG)*](https://www.philschmid.de/fine-tune-embedding-model-for-rag).
