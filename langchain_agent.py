import os
import requests
from dotenv import load_dotenv
from langchain.chains import LLMChain, RetrievalQA
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS

# Load environment variables.
load_dotenv()

# OpenAI and Shopify configuration.
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SHOPIFY_API_KEY = os.getenv("SHOPIFY_API_KEY")
SHOPIFY_PASSWORD = os.getenv("SHOPIFY_PASSWORD")
SHOPIFY_STORE_NAME = os.getenv("SHOPIFY_STORE_NAME")

FAQ_FILES_PATH = os.getenv("FAQ_FILES_PATH", "./faq_files")
DEFAULT_RESPONSE_STYLE = os.getenv("DEFAULT_RESPONSE_STYLE", "professional")

def query_shopify(query: str) -> str:
    """
    Queries the Shopify API to search for product details based on a query string.
    For demonstration, this function makes a GET request to the products endpoint
    and looks for a product whose title matches the query. In a real production
    system, consider using Shopify's GraphQL API and more robust search logic.
    """
    url = f"https://{SHOPIFY_API_KEY}:{SHOPIFY_PASSWORD}@{SHOPIFY_STORE_NAME}.myshopify.com/admin/api/2023-04/products.json"
    try:
        response = requests.get(url)
        response.raise_for_status()
        products = response.json().get("products", [])

        # Naively search for the product in the list by matching the query with product titles.
        for product in products:
            if query.lower() in product.get("title", "").lower():
                # Get the price from the first variant.
                price = product.get("variants", [{}])[0].get("price", "N/A")
                return f"Product '{product.get('title')}' is available at ${price}."
        return "Product not found or out of stock."
    except Exception as e:
        print("Shopify API error:", e)
        return "Error retrieving product details from Shopify."

def get_faq_response(query: str) -> str:
    """
    Uses Retrieval-Augmented Generation (RAG) to fetch an answer from FAQ documents.
    FAQ text files are loaded from the FAQ_FILES_PATH, and a FAISS index is created using
    OpenAI embeddings. Then a RetrievalQA chain answers the query.
    """
    try:
        import os
        faqs = []
        # Load all .txt files from the FAQ directory.
        for filename in os.listdir(FAQ_FILES_PATH):
            if filename.endswith(".txt"):
                with open(os.path.join(FAQ_FILES_PATH, filename), "r", encoding="utf-8") as f:
                    faqs.append(f.read())
        if not faqs:
            return ""

        # Build a FAISS vector store index from the FAQ documents.
        embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
        vector_store = FAISS.from_texts(faqs, embeddings)

        # Create a RetrievalQA chain to find and answer the query.
        qa_chain = RetrievalQA.from_chain_type(
            llm=OpenAI(openai_api_key=OPENAI_API_KEY),
            chain_type="stuff",
            retriever=vector_store.as_retriever()
        )
        result = qa_chain.run(query)
        return result
    except Exception as e:
        print("FAQ retrieval error:", e)
        return ""

def process_customer_query(query: str) -> str:
    """
    Main function that processes a customer query by leveraging both
    Shopify API queries and FAQ retrieval via LangChain.

    1. If the query contains keywords related to products (price, stock, size),
       it calls the Shopify API tool.
    2. It retrieves FAQ information from the FAISS-based retrieval system.
    3. Combines the information using a ReAct-style prompt.
    4. Returns a combined, styled answer.
    """
    # Instantiate the OpenAI language model.
    llm = OpenAI(openai_api_key=OPENAI_API_KEY, temperature=0.7)

    # Simple keyword detection for Shopify-related queries.
    shopify_keywords = ["price", "stock", "product", "size", "in stock", "availability"]
    shopify_response = ""
    if any(keyword in query.lower() for keyword in shopify_keywords):
        shopify_response = query_shopify(query)

    # Retrieve FAQ answer based on the customer query.
    faq_response = (query)

    # Define a prompt template to combine the information.
    prompt_template = PromptTemplate(
        input_variables=["query", "shopify_response", "faq_response", "style"],
        template=("You are a customer support agent for a store. "
                  "A customer asks: '{query}'.\n"
                  "Shopify Info: {shopify_response}\n"
                  "FAQ Info: {faq_response}\n"
                  "Please answer in a {style} tone with clear and friendly information.")
    )

    chain = LLMChain(llm=llm, prompt=prompt_template)
    final_response = chain.run({
        "query": query,
        "shopify_response": shopify_response,
        "faq_response": faq_response,
        "style": DEFAULT_RESPONSE_STYLE
    })

    return final_response
