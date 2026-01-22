
FROM python:3.10-slim

WORKDIR /project


RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt



# COPY app/ ./app/
# COPY models/ ./models/
# COPY faiss_product_rag/ ./faiss_product_rag/
COPY . .


#COPY .env* .


ENV PYTHONPATH=/project
ENV PYTHONUNBUFFERED=1

EXPOSE 8000


CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

