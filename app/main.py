import sys
import os
import joblib
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional
from collections import defaultdict
from dotenv import load_dotenv
import warnings

import app.utils

sys.modules['src'] = app
sys.modules['src.utils'] = app.utils

os.environ["CUDA_VISIBLE_DEVICES"] = ""
warnings.filterwarnings("ignore", message="CUDA initialization")

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
#from langchain.prompts import PromptTemplate
from langchain_core.prompts import PromptTemplate


from .utils import prepare_df_for_inference

load_dotenv()

# Initialize FastAPI app
app = FastAPI(
    title="Marketing Intelligence",
    description="Combined MI and RAG endpoints",
    version="3.0"
)

# ============ MODELS AND DATA (From main.py) ============
MODELS = {}
MODEL_DIR = "models"

def get_model(target_name: str):
    if target_name not in MODELS:
        path = os.path.join(MODEL_DIR, f"{target_name}_model.joblib")
        if os.path.exists(path):
            print(f"Loading {target_name} model...")
            MODELS[target_name] = joblib.load(path)
        else:
            print(f"Warning: Model for {target_name} not found at {path}")
            return None
    return MODELS[target_name]

# ============ RAG COMPONENTS (From rag_app.py) ============
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vectorstore = FAISS.load_local(
    "faiss_product_rag",
    embeddings,
    allow_dangerous_deserialization=True
)

PRODUCT_DOCS = defaultdict(lambda: {"product": [], "review": []})
for doc in vectorstore.docstore._dict.values():
    pid = doc.metadata.get("product_id")
    ctype = doc.metadata.get("chunk_type")
    if pid and ctype:
        PRODUCT_DOCS[pid][ctype].append(doc)

print(f"Indexed {len(PRODUCT_DOCS)} products")

llm = ChatOpenAI(
    model=os.getenv("LLM_MODEL", "gpt-3.5-turbo"),
    temperature=float(os.getenv("LLM_TEMPERATURE", 0.2)),
    api_key=os.getenv("OPENAI_API_KEY")
)

PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are a product assistant.
Answer the question strictly using the context below.
If the answer is not present, say:
"I don't have enough information to answer this."

Context:
{context}

Question:
{question}

Answer:
"""
)

qa_chain = PROMPT | llm

# ============ STARTUP ============
@app.on_event("startup")
def load_all_models():
    for t in ["discounted_price", "rating", "rating_count", "discount_percentage"]:
        get_model(t)

# ============ PYDANTIC MODELS ============
class ProductRequest(BaseModel):
    product_id: Optional[str] = "P001"
    category: Optional[str] = "Electronics"
    about_product: Optional[str] = "Fast charging cable"
    review_title: Optional[str] = ""
    review_content: Optional[str] = ""
    actual_price: Optional[float] = 1000.0
    discounted_price: Optional[float] = 800.0
    discount_percentage: Optional[float] = 20.0
    rating: Optional[float] = 4.0
    rating_count: Optional[float] = 100.0

    class Config:
        extra = "allow"

class CampaignRequest(ProductRequest):
    min_discount: int = Field(10, ge=0, le=90)
    max_discount: int = Field(60, ge=10, le=100)
    step: int = Field(5, gt=0)


class QuestionRequest(BaseModel):
    question: str = "Shall I buy the product at given price?"
    product_id: Optional[str] = ""
    product_name: Optional[str] = "Portronics Konnect"


def detect_intent(question: str):
    q = question.lower()
    print('Q', q)
    if any(w in q for w in ["review", "complaint", "issue", "like", "dislike", "opinion"]):
        return "review"
    return "product"


def resolve_product_id(product_name: str):
    retriever = vectorstore.as_retriever(
        search_kwargs={
            "k": 3,
            "filter": {"chunk_type": "product"}
        }
    )
    docs = retriever.get_relevant_documents(product_name)
    print('DOCS', docs)
    return docs[0].metadata.get("product_id") if docs else None



def resolve_product(product_name: str):
    docs = vectorstore.similarity_search(product_name, k=3)

    if not docs:
        return None, None

    best_doc = docs[0]

    return (
        best_doc.metadata.get("product_id"),
        best_doc.page_content  
    )





def retrieve_docs(
    query: str,
    product_id: str,
    intent: str,
    k: int = 5
):
   
    raw_docs = vectorstore.similarity_search(
        query if query.strip() else "product information",
        k=20  
    )

    
    filtered_docs = [
        d for d in raw_docs
        if d.metadata.get("product_id") == product_id
        and d.metadata.get("chunk_type") == intent
    ]

    return filtered_docs[:k]



def retrieve_docs_faiss(
    product_id: str,
    product_anchor: str,
    intent: str,
    k: int = 5
):
    
    raw_docs = vectorstore.similarity_search(
        product_anchor,
        k=30
    )

    filtered_docs = [
        d for d in raw_docs
        if d.metadata.get("product_id") == product_id
        and d.metadata.get("chunk_type") == intent
    ]

    return filtered_docs[:k]


def retrieve_product_docs(product_id, k=2):
    return PRODUCT_DOCS[product_id]["product"][:k]


def retrieve_review_docs(product_id, k=5):
    return PRODUCT_DOCS[product_id]["review"][:k]

# ============ MARKETING INTELLIGENCE ENDPOINTS (From main.py) ============
@app.post("/predict_price_strategy")
def predict_price_strategy(req: ProductRequest):
    """Predicts the market-standard discounted price based on product features."""
    model = get_model("discounted_price")
    if not model: raise HTTPException(503, "Price model unavailable")

    df = pd.DataFrame([req.dict()])
    df = prepare_df_for_inference(df)
    predicted_price = float(model.predict(df)[0])

    current_price = req.discounted_price or predicted_price
    diff_percent = ((current_price - predicted_price) / predicted_price) * 100

    recommendation = "Hold Price"
    if diff_percent > 20: recommendation = "Lower Price (Overpriced vs Market)"
    if diff_percent < -20: recommendation = "Raise Price (Undervalued)"

    return {
        "predicted_market_price": round(predicted_price, 2),
        "current_price": current_price,
        "deviation_percent": round(diff_percent, 1),
        "recommendation": recommendation
    }

@app.post("/predict_quality_score")
def predict_quality_score(req: ProductRequest):
    """Predicts product rating (quality proxy) from description/features."""
    model = get_model("rating")
    if not model: raise HTTPException(503, "Rating model unavailable")

    df = pd.DataFrame([req.dict()])
    df = prepare_df_for_inference(df)
    rating = float(model.predict(df)[0])

    return {
        "predicted_rating": round(rating, 2),
        "quality_tier": "Premium" if rating > 4.2 else "Standard" if rating > 3.5 else "Needs Improvement"
    }

@app.post("/predict_demand_proxy")
def predict_demand_proxy(req: ProductRequest):
    """Predicts rating_count as a proxy for sales volume/demand."""
    model = get_model("rating_count")
    if not model: raise HTTPException(503, "Demand model (rating_count) unavailable")

    df = pd.DataFrame([req.dict()])
    df = prepare_df_for_inference(df)
    demand = float(model.predict(df)[0])
    demand = max(0, demand)

    return {
        "predicted_demand_score": int(demand),
        "note": "Based on historical review volume for similar products."
    }

@app.post("/optimize_discount_curve")
def optimize_discount_curve(req: CampaignRequest):
    """Simulate revenue at different discount levels to find the 'sweet spot'."""
    demand_model = get_model("rating_count")
    if not demand_model: raise HTTPException(503, "Demand model unavailable")

    results = []
    for d in range(req.min_discount, req.max_discount + 1, req.step):
        base_price = req.actual_price if req.actual_price else 1000
        simulated_price = base_price * (1 - d / 100)
        row_data = req.dict()
        row_data["discount_percentage"] = d
        row_data["discounted_price"] = simulated_price
        df = pd.DataFrame([row_data])
        df = prepare_df_for_inference(df)
        pred_demand = float(demand_model.predict(df)[0])
        pred_demand = max(1, pred_demand)
        revenue = simulated_price * pred_demand
        results.append({
            "discount_percent": d,
            "simulated_price": round(simulated_price, 2),
            "predicted_demand_units": int(pred_demand),
            "projected_revenue": round(revenue, 2)
        })

    best_option = max(results, key=lambda x: x["projected_revenue"])

    return {
        "optimal_discount": best_option["discount_percent"],
        "max_revenue_projected": best_option["projected_revenue"],
        "simulation_curve": results
    }

@app.post("/estimate_conversion_potential")
def estimate_conversion_potential(req: ProductRequest):
    """Combines Quality (Rating) and Popularity (Count) into a single score."""
    r_model = get_model("rating")
    d_model = get_model("rating_count")

    if not r_model or not d_model:
        raise HTTPException(503, "Models missing")

    df = pd.DataFrame([req.dict()])
    df = prepare_df_for_inference(df)
    pred_rating = float(r_model.predict(df)[0])
    pred_demand = float(d_model.predict(df)[0])
    pred_demand = max(0, pred_demand)
    score = pred_rating * np.log1p(pred_demand)

    return {
        "conversion_potential_score": round(score, 2),
        "predicted_rating": round(pred_rating, 1),
        "predicted_volume": int(pred_demand),
        "market_segment": (
            "Star Product (High Vol/High Rating)" if score > 25 and pred_rating > 4.0 else
            "Niche Gem (Low Vol/High Rating)" if pred_rating > 4.2 else
            "Mass Market (High Vol/Avg Rating)" if pred_demand > 500 else
            "Risk (Low Vol/Low Rating)"
        )
    }

# ============ RAG ENDPOINT (From rag_app.py) ============
@app.post("/answer_question")
def answer_question(req: QuestionRequest):


    product_id = req.product_id

    if not product_id and req.product_name:
        product_id, _ = resolve_product(req.product_name)

        if not product_id:
            return {
                "answer": "I couldn't find a matching product.",
                "sources": []
            }


    intent = detect_intent(req.question)

   
    if intent == "product":
        docs = retrieve_product_docs(product_id)
    else:
        docs = retrieve_review_docs(product_id)

    if not docs:
        return {
            "answer": "I don't have enough information to answer this.",
            "sources": []
        }

  
    context = "\n\n".join(d.page_content for d in docs)

    response = qa_chain.invoke(
        {"context": context, "question": req.question}
    )

    return {
        "answer": response.content,
        "resolved_product_id": product_id,
        "sources": [
            {
                "product_id": d.metadata.get("product_id"),
                "chunk_type": d.metadata.get("chunk_type"),
                "review_id": d.metadata.get("review_id")
            }
            for d in docs
        ]
    }



# ============ HOME ENDPOINT ============
@app.get("/")
def home():
    return {
        "endpoints": [
            "/predict_price_strategy",
            "/predict_quality_score",
            "/predict_demand_proxy",
            "/optimize_discount_curve",
            "/estimate_conversion_potential",
            "/answer_question"
        ]
    }
