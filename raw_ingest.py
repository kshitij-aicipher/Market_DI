import pandas as pd
from langchain.schema import Document
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS


df = pd.read_csv("data/raw/amazon_products.csv")
df = df.fillna("")


def split_field(value):
    return [v.strip() for v in value.split(",") if v.strip()]


product_docs = []

for product_id, group in df.groupby("product_id"):
    row = group.iloc[0]

    product_text = f"""
Product Name: {row['product_name']}
Category: {row['category']}
Actual Price: {row['actual_price']}
Discounted Price: {row['discounted_price']}
Discount Percentage: {row['discount_percentage']}
Rating: {row['rating']}
Rating Count: {row['rating_count']}
Description: {row['about_product']}
Product Link: {row['product_link']}
"""

    product_docs.append(
        Document(
            page_content=product_text.strip(),
            metadata={
                "product_id": product_id,
                "chunk_type": "product",
                "category": row["category"]
            }
        )
    )


review_docs = []

for _, row in df.iterrows():
    user_ids = split_field(row["user_id"])
    user_names = split_field(row["user_name"])
    review_ids = split_field(row["review_id"])
    review_titles = split_field(row["review_title"])
    review_contents = split_field(row["review_content"])

    
    for uid, uname, rid, title, content in zip(
        user_ids,
        user_names,
        review_ids,
        review_titles,
        review_contents
    ):
        review_text = f"""
Product Name: {row['product_name']}
Review Title: {title}
Review Content: {content}
"""

        review_docs.append(
            Document(
                page_content=review_text.strip(),
                metadata={
                    "product_id": row["product_id"],
                    "review_id": rid,
                    "user_name": uname,
                    "chunk_type": "review",
                    "rating": row["rating"]
                }
            )
        )


documents = product_docs + review_docs

print(f"Product documents: {len(product_docs)}")
print(f"Review documents: {len(review_docs)}")
print(f"Total documents: {len(documents)}")


embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


vectorstore = FAISS.from_documents(
    documents=documents,
    embedding=embeddings
)

vectorstore.save_local("faiss_product_rag")

print("FAISS index created and saved successfully.")



# Product documents: 1351
# Review documents: 11437
# Total documents: 12788