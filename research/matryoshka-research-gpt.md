# Matryoshka Embeddings in Modern Semantic Models (July 2025 Update)

**Matryoshka Representation Learning (MRL)** is a training technique that produces nested embeddings—vectors that remain useful even after truncation. In these models, the most important semantic information is packed into the first dimensions. Drop the later components and you still retain most of the meaning. The payoff: flexible trade-offs between accuracy and efficiency. Smaller vectors mean less storage and faster similarity computations; larger ones preserve full performance. This enables workflows like shortlist-and-rerank, where initial retrieval uses compact embeddings, followed by refinement with full-size vectors.

## English Monolingual Models with MRL

* **Sentence Transformers Matryoshka Models:** The Sentence-Transformers library added MRL support in 2024, simplifying fine-tuning for variable-length embeddings. One example is an MPNet model trained on NLI data using a combined Matryoshka + MultipleNegativesRanking loss. This English model (`tomaarsen/mpnet-base-nli-matryoshka`) outputs 768-dimensional embeddings truncatable to 512, 256, 128, etc., with minimal performance loss. At ~8% of original size, it retains ~98.4% of semantic similarity performance—better than standard models at the same reduced size. Similar approaches have produced biomedical and financial-domain variants based on PubMedBERT and BAAI’s BGE, respectively.

* **Nested Embeddings in Other Languages:** MRL has expanded beyond English. Researchers in Arabic NLP applied it to models like GATE-Arabert, achieving strong STS17 scores while supporting smaller vector sizes. This confirms MRL’s benefits aren’t language-specific.

* **Multi-Modal Matryoshka (CLIP-based):** Marqo AI extended MRL to CLIP-derived models (called **GCL**) for joint image-text embeddings. Training with Matryoshka loss (e.g., dimensions {512, 256, 128, 64}) showed truncated vectors maintain high retrieval performance. Unlike non-MRL CLIP models, which degrade quickly when truncated, GCL’s drop in normalized DCG was negligible. A single MRL-trained model can serve both full and reduced embeddings effectively, showing MRL works across modalities.

**General vs. Search-Focused Models:** Many English MRL models are trained on general semantic tasks (NLI, STS) and work well for clustering, search, and classification. Others target information retrieval directly, using contrastive datasets. For instance, `pubmedbert-matryoshka` is built for biomedical literature search, and the financial BGE variant includes IR-specific evaluations. In practice, MRL adds value to both paths—general models gain speed for approximate matching, while search-focused models gain adjustable accuracy at query time. The usual pattern: index and query with small vectors, then rerank top hits using the full vector. Tools like Vespa and Weaviate support this workflow for OpenAI’s MRL embeddings.

## Multilingual and Commercial Models with MRL

* **OpenAI’s Embedding Models (text-embedding-3):** In January 2024, OpenAI launched `text-embedding-3-large` (3072 dims) and `text-embedding-3-small` (768 dims), both supporting dynamic dimensionality via API. Internally, they use MRL. A 256-dim version of *3-large* outperforms the older ada-002 model (1536 dims) on MTEB. These models are multilingual, with MIRACL scores roughly double those of ada-002. MRL embeddings from OpenAI are now supported in vector databases like Vespa and Weaviate, enabling adaptive retrieval: coarse search with smaller dims, refinement with larger.

* **Alibaba’s GTE Family:** Alibaba’s **GTE-multilingual-base** is an open multilingual model explicitly trained with MRL. It outputs 768-dim embeddings but supports any size from 128 to 768 at inference. Trained on a large multilingual corpus, it delivers state-of-the-art retrieval performance. Alibaba also offers English-optimized and instruction-tuned variants (e.g., gte-large-en, gte-Qwen), though the encoder-only base model stands out for its resizable embedding feature. MRL is not just a research idea—it’s part of production-ready multilingual systems. Note: BAAI’s **BGE-M3** supports multi-language and multi-granularity, but its “multi-granularity” refers to input handling and multi-vector outputs, not MRL-style dimensionality reduction.

* **Nomic’s Resizable Embeddings:** Nomic AI introduced open-source models built heavily on MRL. **Nomic Embed v1.5** (Feb 2024) was a 768-dim English model trained with Matryoshka loss, supporting output sizes as low as 64 dims—even binary via sign bit. It outperformed OpenAI’s ada-002 at equivalent or smaller sizes. In 2025, Nomic released **Embed Text V2**, a multilingual model using Mixture-of-Experts and MRL. It supports truncation from 768 to 256 dims with no measurable quality loss on BEIR/MIRACL benchmarks. The model weights and training code are open-sourced, making it one of the most accessible high-performance MRL models. Designed for semantic search and retrieval, it supports flexible deployment without retraining.

* **Voyage-3-large (Multilingual SOTA):** In 2025, Voyage AI launched `voyage-3-large`, a multilingual embedding model combining MRL with quantization-aware training. It outputs 2048-dim vectors but supports 1024, 512, and 256 dims from the same model. Even at 256 dims, it beats OpenAI’s 3072-dim model by ~11% in retrieval tasks. With aggressive quantization (8-bit or binary), storage can be reduced up to 200× while maintaining superior accuracy. For example, an int8 1024-dim vector performs within 0.3% of the full 2048-dim float vector. While commercial, Voyage-3 shows how MRL is being used to outpace established models in real-world applications.

In summary, as of mid-2025, many modern embedding models use MRL, even if not explicitly named. Open-source examples include Sentence-Transformers models (English and Arabic), domain-specific embeddings (PubMedBERT, financial BGE), and multilingual models like Alibaba’s GTE. On the commercial side, OpenAI and Voyage AI have integrated MRL for scalable dimensionality. The core idea is consistent: train embeddings to be coarse-to-fine. This enables adaptive workflows in clustering, similarity search, and retrieval—bulk operations run on small vectors, final steps use full detail. MRL has graduated from research curiosity to practical standard.

**Sources:**

* Kusupati et al., *“Matryoshka Representation Learning”*, NeurIPS 2022  
* Hugging Face Blog – *“Introduction to Matryoshka Embedding Models”*, Feb 2024  
* Sentence-Transformers Docs – *“Matryoshka Embeddings”*  
* Nomic AI Blog – *“Unboxing Nomic Embed v1.5”*, Feb 2024  
* Nomic AI Blog – *“Embed Text V2”*, 2025  
* OpenAI Product Update – *“New embedding models with lower pricing”*, Jan 2024  
* Alibaba GTE Model Card – *gte-multilingual-base*, 2023  
* Milvus Blog – *“Matryoshka Embeddings: Detail at Multiple Scales”*, 2024  
* Marqo Blog – *“MRL with CLIP for Multimodal Retrieval”*, 2025  
* Voyage AI Blog – *“voyage-3-large: SOTA general-purpose embedding model”*, Jan 2025