
import os
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

from langchain.tools import tool
from langchain.agents import create_agent


# Load API key
load_dotenv()


# 1. Load PDF
loader = PyPDFLoader("Movies.pdf")
documents = loader.load()

print("PDF pages:", len(documents))


# 2. Split documents into chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

chunks = splitter.split_documents(documents)

print("Total chunks:", len(chunks))


# 3. Create Hugging Face embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# 4. Store embeddings in ChromaDB
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./chroma_db"
)

print("Documents stored successfully!")


# 5. Create retriever
retriever = vectorstore.as_retriever(
    search_kwargs={"k": 4}
)


# 6. Create RAG tool
@tool
def search_movie_documents(question: str) -> str:
    """Search the movie PDF and return relevant movie information."""

    results = retriever.invoke(question)

    if not results:
        return "No relevant movie information found."

    context = "\n\n".join(
        [
            f"Page {doc.metadata.get('page', 0) + 1}:\n{doc.page_content}"
            for doc in results
        ]
    )

    return context



# 7. Create Groq LLM
llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0
)



# 8. Create AI Agent
agent = create_agent(
    model=llm,
    tools=[search_movie_documents],
    system_prompt="You are a helpful Movie Information Assistant. Use the search_movie_documents tool before answering movie questions."
)


# 9. Chat with Agent
print("\nMovie RAG Agent is ready!")

while True:

    question = input("\nAsk your movie question (type exit to stop): ")

    if question.lower() == "exit":
        print("Goodbye!")
        break

    response = agent.invoke(
        {
            "messages": [
                {"role": "user", "content": question}
            ]
        }
    )

    print("\nAI Answer:")

    print(response["messages"][-1].content)
