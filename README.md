# Fashion Image Similarity Search

A visual search engine for fashion products. Upload a clothing image and instantly find the most similar items from the catalog — ranked by visual similarity, filtered by metadata, and reranked with keyword matching.

Built with CLIP embeddings, cosine similarity search, and a Streamlit UI.

---

## Demo

**Image search** — upload any clothing photo, get back the top-k most visually similar products:

```
Query: navy blue check shirt
→ #1  Turtle Check Men Navy Blue Shirt       (score: 1.000)
→ #2  Indigo Nation Men Checks Shirt         (score: 0.875)
→ #3  Locomotive Men Check Red Shirt         (score: 0.621)
```

**Text search** — describe what you're looking for in plain English:

```
python demo.py --text "red floral summer dress" --gender Women
```

---

## How It Works

```
Query Image
    │
    ▼
CLIP ViT-B/32 Image Encoder       ← pretrained on 400M image-text pairs
    │
    ▼
512-dim embedding vector (L2 normalised)
    │
    ▼
Metadata Filter                    ← narrow by gender / category / colour / season
    │
    ▼
Cosine Similarity Search           ← dot product over catalog embeddings (NumPy)
    │
    ▼
Reranker                           ← blends CLIP score + keyword match score
    │
    ▼
Top-K Results  (image path · name · category · colour · score breakdown)
```

**No training required.** CLIP was pretrained by OpenAI; we use it purely for inference. Embeddings are computed once and cached to disk — subsequent searches are instant.

---

## Features

- **CLIP-powered visual search** — understands texture, colour, cut, and style
- **Text search** — find items by natural language description ("oversized grey hoodie")
- **Metadata filtering** — hard constraints on gender, category, subcategory, article type, colour, season
- **Reranking** — blends CLIP visual score (70%) with keyword match score (30%) for better precision
- **Embedding cache** — catalog embedded once, reloaded instantly on subsequent runs
- **Streamlit UI** — browser-based interface with filter sidebar and score breakdown
- **CLI demo** — scriptable command-line interface for quick testing

---

## Tech Stack

| Component | Technology |
|---|---|
| Embedding model | OpenAI CLIP ViT-B/32 |
| Deep learning | PyTorch |
| Similarity search | NumPy (cosine similarity) |
| Dataset | Myntra Fashion Dataset (44k products) |
| Frontend | Streamlit |
| Language | Python 3.10+ |

---

## Project Structure

```
├── src/
│   ├── config.py         # paths, model name, search hyperparameters
│   ├── data_loader.py    # reads styles.csv, resolves image paths
│   ├── embeddings.py     # CLIP encoder + disk cache (.npz)
│   └── search.py         # filter → retrieve → rerank pipeline
├── app.py                # Streamlit UI
├── demo.py               # CLI entry point
├── data/
│   └── metadata.json     # sample product catalog schema
└── requirements.txt
```

---

## Setup

```bash
# Clone the repo
git clone https://github.com/dongyansun/Similarity-search-project.git
cd Similarity-search-project

# Install dependencies
pip install -r requirements.txt

# Download the Myntra dataset and place it as:
#   archive (1)/images/{id}.jpg
#   archive (1)/styles.csv
# Dataset: https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-dataset
```

---

## Usage

### Streamlit UI

```bash
streamlit run app.py
```

Opens at `http://localhost:8501`. Upload an image or type a description, apply filters from the sidebar, toggle reranking.

### CLI

```bash
# Image search
python demo.py --query "path/to/your/image.jpg"

# Image search with filter and rerank hint
python demo.py --query "path/to/image.jpg" --gender Men --rerank "navy blue shirt"

# Text search (reranking on by default)
python demo.py --text "floral summer dress" --gender Women

# Change number of results
python demo.py --query "path/to/image.jpg" --top_k 10

# Re-embed catalog after adding new products
python demo.py --query "path/to/image.jpg" --rebuild
```

### Filter options

| Flag | Field | Example values |
|---|---|---|
| `--gender` | gender | Men, Women, Boys, Girls, Unisex |
| `--category` | masterCategory | Apparel, Accessories, Footwear |
| `--subcat` | subCategory | Topwear, Bottomwear, Shoes |
| `--type` | articleType | Shirts, Jeans, Dresses, Watches |
| `--colour` | baseColour | Navy Blue, Red, Black, White |
| `--season` | season | Summer, Winter, Fall, Spring |

---

## Model Options

Edit `src/config.py` to change the CLIP model:

```python
CLIP_MODEL_NAME = "ViT-B/32"   # fast, good baseline
CLIP_MODEL_NAME = "ViT-B/16"   # better accuracy, same size
CLIP_MODEL_NAME = "ViT-L/14"   # best accuracy, ~3x slower to embed
```

After switching models, delete `data/embeddings_cache.npz` and run with `--rebuild`.
