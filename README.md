<center>

# **Smart News Crawl & QA**

</center>

---

## NewsScraper

A Python-based asynchronous web scraper for extracting news articles from a specified website using `crawl4ai` and `BeautifulSoup`.

### Steps / Operations Performed

1. **Input Website URL**  
   Prompt the user to enter the base URL of a news website to scrape.

2. **Input Optional Keywords**  
   Prompt the user to input comma-separated keywords for article content filtering (optional).

3. **Initialize Scraper**  
   - Set the base URL and extract its domain.
   - Define the maximum number of articles to scrape.
   - Optionally apply keyword-based content filtering.

4. **Start Crawling**  
   - Use `AsyncWebCrawler` with headless browser configuration.
   - Begin from the given base URL and follow internal links recursively.

5. **Extract and Parse Content**  
   - For each page visited:
     - Parse article title, content, and publication time.
     - Filter out irrelevant articles (e.g., content too short or not matching keywords).
     - Avoid revisiting URLs.
     - Limit scraping to the configured number of articles.

6. **Follow Internal Links**  
   - Extract `<a>` tags and resolve relative URLs.
   - Continue crawling within the same domain.

7. **Save Results to JSON**  
   Save all successfully scraped articles into a file named `scraped_news.json`.

8. **Completion Message**  
   Display the total number of articles saved.

### Notes

- Scraping is performed asynchronously for efficiency.
- Only internal links within the same domain are followed.
- Filtering by keywords is case-insensitive and optional.
- HTML is parsed using `BeautifulSoup`.

-------------------------------------------------------------------------------------------------------

## News Article Processing and Agentic QA Pipeline

### Steps / Operations Performed

1. **Load environment variables**  
   Load OpenAI API key and other settings from `.env` file.

2. **Load scraped JSON news articles**  
   Read JSON file of news articles and format each as a string combining title, timestamp, and content.

3. **Split documents into chunks**  
   Use `RecursiveCharacterTextSplitter` to split long article strings into manageable overlapping chunks (chunk size 500, overlap 100).

4. **Create vectorstore (FAISS) and build retriever**  
   - Generate embeddings using OpenAI embeddings.  
   - Build a FAISS index from the document chunks.  
   - Create a retriever to return top 4 relevant chunks for queries.

5. **Setup Agentic Retrieval-Augmented Generation (RAG) QA Chain**  
   Initialize a ChatOpenAI model (`gpt-4o`) with zero temperature for deterministic answers, combined with the retriever to answer queries in an agentic manner.

6. **Agentic News Summarizer**  
   - Take top 10 news articles, join their content.  
   - Use a prompt template to summarize key highlights into three paragraphs with an agentic focus on context and logical grouping.  
   - Run the prompt through the ChatOpenAI model with moderate temperature (0.5).

7. **Agentic Direct LLM answer with JSON context and output**  
   - Convert top retrieved documents' content to JSON format.  
   - Construct a prompt instructing the agentic LLM to produce a structured JSON response with `answer`, `source_titles`, and `confidence_score`.  
   - Use the ChatOpenAI model to generate the precise JSON answer agentically using only the provided context.

8. **Interactive agentic mode selection**  
   User chooses from three agentic modes:  
   - **[1] Agentic Summarizer:** Generate an agentic summary of latest news articles.  
   - **[2] Agentic Q&A:** Ask questions with the agentic RAG QA chain for free-text answers.  
   - **[3] Agentic JSON Context QA:** Ask questions and receive structured JSON answers based on retrieved articles with agentic reasoning.

9. **Loop for user agentic queries**  
   For modes 2 and 3, continuously accept user input until exit command is given, enabling agentic interaction.

---

This pipeline enables agentic interaction with news data from raw ingestion through embedding, retrieval, and structured or summarization-based response generation.

### License
\033[93mMSIS, Manipal\033[0m
