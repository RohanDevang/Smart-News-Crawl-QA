import json
import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.docstore.document import Document
from langchain_core.runnables import RunnableLambda, RunnableMap
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

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
    return vectorstore.as_retriever(search_kwargs={"k": 5})

# Step 4: Setup RAG QA Chain
def build_qa_chain(retriever):
    openai_api_key = os.getenv("OPENAI_API_KEY")
    llm = ChatOpenAI(model_name="gpt-4o", temperature=0, openai_api_key=openai_api_key)
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
    return qa_chain

# Step 5: News Summarizer Agent (migrated from deprecated chain)
def summarize_news(docs):
    openai_api_key = os.getenv("OPENAI_API_KEY")
    llm = ChatOpenAI(model_name="gpt-4o", temperature = 0.5, openai_api_key=openai_api_key)
    input_docs = docs[:10]  # top 10
    joined_content = "\n\n".join(input_docs)

    prompt_template = PromptTemplate.from_template(
        """
        Summarize the key highlights from the following news articles in three well-structured paragraphs.
        Focus on the most important developments, provide context where necessary, and group related points logically.

        {documents}

        """
    )

    chain = (
        {"documents": lambda _: joined_content}
        | prompt_template
        | llm
        | StrOutputParser()
    )

    summary = chain.invoke({})
    return summary

# Step 6: Direct LLM answer with JSON context and output

def generate_precise_answer(llm, docs, question):
    context_json = json.dumps([doc.page_content for doc in docs], ensure_ascii=False, indent=2)
    prompt = f"""
                You are an expert news assistant. Given the following news articles in JSON format, 
                answer the user's question as a structured JSON object with 'answer', 'source_titles', and 'confidence_score'.
                Only use the information from the provided articles. If the answer cannot be found, say "I don't know.".

News Articles (JSON):
{context_json}

Question: {question}

Provide your response in the following JSON format:
{{
  "answer": "<concise answer>",
  "source_titles": ["<title1>", "<title2>"],
  "confidence_score": <0 to 1>
}}
"""
    response = llm.invoke(prompt)
    return response.content if hasattr(response, 'content') else response

if __name__ == '__main__':
    json_file = 'scraped_news.json'

    print("\n🔄 Loading and preparing news data...")
    raw_docs = load_news_articles(json_file)
    split_docs = split_documents(raw_docs)

    print("🧠 Creating Retriever with top k = 5...")
    retriever = create_vectorstore(split_docs)

    print("🤖 Setting up \033[92mRAG\033[0m pipeline with LLM...")
    openai_api_key = os.getenv("OPENAI_API_KEY")
    llm = ChatOpenAI(model_name="gpt-4o", temperature=0, openai_api_key=openai_api_key)
    qa = build_qa_chain(retriever)

    while True:
        print("\n🤖 Choose mode: [\033[1;92m1\033[0m] Agentic Summarizer, "
              "[\033[1;92m2\033[0m] Agentic Q&A mode, "
              "[\033[1;92m3\033[0m] Agentic JSON Context QA, "
              "[\033[1;91m0\033[0m] Exit")
        mode = input("Enter your choice (1, 2, 3 or 0 to exit): ").strip().lower()

        if mode in ['0', 'exit', 'quit']:
            print("\n👋 Exiting the program. Goodbye!\n")
            break

        elif mode == '1':
            print("\n📰 Generating summary of latest news...")
            summary = summarize_news(raw_docs)
            print(f"\n📝 \033[93mNews Summary:\033[0m \n {summary}\n")

        elif mode == '2':
            print("\033[1;92mAgentic Q&A mode\033[0m")
            while True:
                query = input("\n\033[93mAsk a question about the 'NEWS' or type 'quit': \033[0m")
                if query.lower() in ['exit', 'quit']:
                    break
                result = qa.invoke({"query": query})
                print(f"\n\033[92mAnswer:\033[0m {result['result']}")

        elif mode == '3':
            print("\033[1;92mAgentic JSON Context QA\033[0m")
            while True:
                query = input("\n\033[93mAsk a question about the 'NEWS' for JSON or type 'quit': \033[0m")
                if query.lower() in ['exit', 'quit']:
                    break
                top_docs = retriever.invoke(query)
                answer = generate_precise_answer(llm, top_docs, query)
                print(f"\n🧾 \033[93mJSON Answer:\033[0m\n\n{answer}")
        else:
            print("\n❌ Invalid choice. Please enter 1, 2, 3, or 0 to exit.")
