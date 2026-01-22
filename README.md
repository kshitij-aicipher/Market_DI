# Marketing Intelligence & Product RAG Assistant

Have created a unified FastAPI application combining **Marketing Intelligence (MI)** predictive models with a **Retrieval-Augmented Generation (RAG)** product assistant. This system predicts optimal pricing strategies and answers customer questions using vector search over product reviews.

## 

### 1. Marketing Intelligence (MI)
- **Price Strategy Prediction**: Suggests "Hold", "Raise", or "Lower" price based on market value.
- **Dynamic Pricing Simulation**: Simulates revenue outcomes for discount campaigns (10-60%).
- **Quality & Demand Scoring**: Predicts product ratings and sales volume potential.
- **Conversion Potential**: Classifies products into segments like "Star Product" or "Niche Gem".

### 2. RAG Product Assistant
- **Context-Aware Q&A**: Answers user questions using indexed product details and customer reviews.
- **Intent Detection**: Automatically routes queries to product specs or review sentiment.
- **Source Citation**: Returns exact source chunks (product vs. review) used for answers.

---

##  Project Structure

```bash
├── app/
│   ├── main.py          # FastAPI application
│   └── utils.py         # Data cleaning & preprocessing
├── models/              # Pre-trained ML models 
├── faiss_product_rag/   # FAISS vector store for RAG
├── data/                # Raw data 
└── training/            # Training scripts 
```

---

## Installation & Local Setup

### Prerequisites
- Python 3.8+ (Recommended 3.8 to match model training)
- Docker (optional)

### 1. Clone & Install Dependencies
```bash
git clone <repo-url>
cd <repo-name>

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install requirements
pip install -r requirements.txt
```

### 2. Environment Configuration
Create a `.env` file in the root directory:
```ini
OPENAI_API_KEY=sk-your-key-here
LLM_MODEL=gpt-3.5-turbo(or any)
LLM_TEMPERATURE=0.2
```

### 3. Run Application
```bash
uvicorn app.main:app --reload
```
Access Swagger UI at: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)


### Run Container
```bash
docker run -p 8000:8000 --env-file .env mi-rag-app
```

---

## API Endpoints

### Marketing Intelligence
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/predict_price_strategy` | POST | Compare current price vs. predicted market price. |
| `/optimize_discount_curve` | POST | Find the revenue-maximizing discount percentage. |
| `/estimate_conversion_potential` | POST | Score product viability (Rating × Demand). |

### RAG Assistant
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/answer_question` | POST | Ask questions about products (e.g., "Is the battery good?"). |

---

## Example Usage

**Request (`/answer_question`):**
```json
{
  "question": "What do people say about the sound quality?",
  "product_name": "Sony WH-1000XM4"
}
```

**Response:**
```json
{
  "answer": "Users generally praise the noise cancellation.",
  "resolved_product_id": "P001",
  "sources": [{"chunk_type": "review", "review_id": "R123"}]
}
```


##
Have used FAISS for lightweight and on the fly manipulations.
Very lightweight models as dataset was not huge.


{
  "discounted_price": {
    "rmse": 1517.0224957995272,
    "mae": 630.7298939260008,
    "r2": 0.9544977043340171
  },
  "actual_price": {
    "rmse": 2574.213938598804,
    "mae": 1010.0269116629394,
    "r2": 0.9450551115833026
  },
  "discount_percentage": {
    "rmse": 13.72243689355659,
    "mae": 10.18308342590406,
    "r2": 0.5180995066418304
  },
  "rating": {
    "rmse": 0.21514603437852803,
    "mae": 0.15459570480625512,
    "r2": 0.403480446812175
  },
  "rating_count": {
    "rmse": 241.69408659779276,
    "mae": 186.2744025194228,
    "r2": 0.09850703322563292
  }
}

## Potential Improvements
There are few parts, which I have skipped due to time constraint on my end (Focus was on business approach to given problem).
These are pointers which I'd like to mention, would've made it more robust and end to end POC worthy.

1. APIs - Could use JWT token for restricting unauthorised users. which will check for token on each call.
        - Giving access to APIs based on user type. For ex - 
            'admin' will have access to price strategy and dynamic pricing
            'consumer' can ask questions regarding product suggestions/reviews
        - Logs for each API call, storing username who triggered at what time.

2. Model  - Integration of MLflow/kubeflow to keep and track the model versions along with model summary. Triggering retraining when detectd model drift based on provided parameters, also considering we have significant additoinal/improved data available.
          - Serve models through APIs (For scaling) and maintain featurestore.
          - As I took only 1500 datapoints, didn't work much on improving accuracy. With more data, it can be done.

3. RAG  - Can add support for more LLM APIs. Currently supports openai.
        - Haven't added any rerankers or much manipulation for improved accuracy. can be optimized in future.

4.  - Can create **Streamlit/Gradio** UI for better experience. Currently working on basic swagger UI.
    - Haven't added docker-compose as currently running everything in single app.
    - Haven't added versions in requirements
    - Dockerfile is not optimized and haven't included folders of sentence-transformers, so it may attempt to download when creating container.
