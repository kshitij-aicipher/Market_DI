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


## Model Evaluation Metrics

The table below summarizes the performance of trained regression models on the validation dataset.

**Metrics used:**
- **RMSE** – Root Mean Squared Error  
- **MAE** – Mean Absolute Error  
- **R²** – Coefficient of Determination  

| Target Variable        | RMSE     | MAE      | R² Score |
|------------------------|----------|----------|----------|
| Discounted Price       | 1517.02  | 630.73   | 0.9545   |
| Actual Price           | 2574.21  | 1010.03  | 0.9451   |
| Discount Percentage    | 13.72    | 10.18    | 0.5181   |
| Rating                 | 0.2151   | 0.1546   | 0.4035   |
| Rating Count           | 241.69   | 186.27   | 0.0985   |

### Notes

- Price prediction models (`actual_price`, `discounted_price`) achieve high accuracy with **R² > 0.94**.
- Discount percentage and rating show moderate predictive power.
- Rating count is highly noisy and therefore has lower explanatory power.

### Raw Evaluation Output

```json
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
```


## RAG Evaluation

RAG performance was evaluated using a manually created evaluation dataset.

**Evaluation Metrics:**
1. **Grounding (0.62)**  
   Measured by checking token overlap between the generated answer and the retrieved context.  
   *(Threshold kept intentionally low.)*

2. **Factuality (0.68)**  
   Evaluated using an LLM-as-a-judge approach.

**Observed Failure Cases:**
- Reviews containing conflicting opinions
- Generic or subjective questions requiring detailed judgment
- Product descriptions lacking sufficient detail

---

## Potential Improvements

Due to time constraints and a focus on demonstrating the **business-oriented approach**, some components were intentionally scoped down.  
Below are areas that would make the solution more **robust and production-ready**.

---

### API Layer

- Implement **JWT-based authentication** to restrict unauthorized access.
- Enable **role-based access control (RBAC)**:
  - `admin` → access to price strategy and dynamic pricing APIs
  - `consumer` → access to product suggestions and review-based queries
- Add **API request logging**, capturing:
  - Username
  - Timestamp
  - Endpoint accessed

---

### Model Lifecycle & MLOps

- Integrate **MLflow / Kubeflow** for:
  - Model versioning
  - Experiment tracking
  - Storing model summaries and metrics
- Detect **model drift** and trigger automated retraining based on:
  - Statistical drift
  - Availability of significant new or improved data
- Serve models via **dedicated inference APIs** for scalability.
- Introduce a **feature store** for consistent training and inference pipelines.
- Limited accuracy optimization was performed due to using only ~1500 data points.  
  Performance can be significantly improved with larger datasets.

---

### RAG Enhancements

- Add support for **multiple LLM providers** (currently supports OpenAI only).
- Introduce **rerankers** and advanced retrieval strategies to improve answer relevance.
- Optimize prompt engineering and context selection logic.

---

### Platform & Deployment

- Build a **Streamlit / Gradio UI** for better user experience  
  *(currently using basic Swagger UI)*.
- Add **docker-compose** for multi-service deployment (API, vector store, model service).
- Pin **dependency versions** in `requirements.txt`.
- Optimize the Dockerfile:
  - Use multi-stage builds
  - Reduce image size
  - Pre-bundle `sentence-transformers` models to avoid runtime downloads

