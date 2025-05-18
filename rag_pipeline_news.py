import json
import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Load environment variables from .env file
load_dotenv()

# Step 1: Load scraped JSON news articles
def load_news_articles(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    docs = []
    for item in data:
        content = f"Title: {item['title']}\nDate: {item.get('timestamp', 'N/A')}\nContent: {item['content']}"
        docs.append(content)
    return docs

# Step 2: Split documents into chunks for better embedding
def split_documents(docs):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    return splitter.create_documents(docs)

# Step 3: Create vectorstore (FAISS) and build retriever
def create_vectorstore(docs):
    openai_api_key = os.getenv("OPENAI_API_KEY")
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    vectorstore = FAISS.from_documents(docs, embeddings)
    # Set retriever to return top 4 documents
    return vectorstore.as_retriever(search_kwargs={'k': 4})

# Step 4: Generate a precise answer from top-k docs using LLM
def generate_precise_answer(llm, docs, question):
    # Format the docs as a JSON array
    context_json = json.dumps([doc.page_content for doc in docs], ensure_ascii=False, indent=2)
    prompt = f"""
                You are an expert news assistant. Given the following news articles in JSON format, 
                answer the user's question as precisely and concisely as possible.
                Only use the information from the provided articles. If the answer cannot be found, say "I don't know."

News Articles (JSON):
{context_json}

Question: {question}

Answer:
"""
    response = llm.invoke(prompt)
    return response.content if hasattr(response, 'content') else response

if __name__ == '__main__':
    json_file = 'scraped_news.json'

    print("\n🔄 Loading and preparing news data...")
    raw_docs = load_news_articles(json_file)
    split_docs = split_documents(raw_docs)

    print("🧠 Creating retriever...")
    retriever = create_vectorstore(split_docs)

    print("🤖 Setting up LLM for precise QA...")
    print("")
    openai_api_key = os.getenv("OPENAI_API_KEY")
    llm = ChatOpenAI(model_name="gpt-4o", temperature = 0, openai_api_key=openai_api_key)

    while True:
        print("\n\033[93mAsk a question about the NEWS:\033[0m", end=' ')
        query = input()
        if query.lower() in ['exit', 'quit']:
            break

        # Retrieve top 4 docs
        top_docs = retriever.invoke(query)
        
        # Generate answer
        answer = generate_precise_answer(llm, top_docs, query)
        print(f"\n\033[92mAnswer:\033[0m {answer}")

