# Club Information RAG Agent

This repository contains a simple yet effective Retrieval-Augmented Generation (RAG) agent designed to answer questions about a specific document. Built with Streamlit and powered by OpenAI/OpenRouter models, this application serves as a conversational interface to a knowledge base, in this case, a document about a student developer club.

The agent ensures that answers are strictly derived from the provided text, telling the user when information is not available in the source document.

## Features

- **Retrieval-Augmented Generation (RAG):** Answers user questions using information retrieved from a local text file.
- **Dynamic Indexing:** Builds a vector index from the provided `club_data.txt` or a user-uploaded file on the fly.
- **Hybrid Search:** Implements a two-stage retrieval process:
    1.  **Vector Search:** Fast retrieval of semantically similar text chunks using cosine similarity.
    2.  **Re-ranking:** Improves relevance by re-ordering results based on a combined score of semantic similarity and keyword overlap.
- **Interactive UI:** A simple and clean user interface built with Streamlit for asking questions and managing the data source.
- **Source-Grounded Answers:** The LLM is prompted to answer *only* using the retrieved context, preventing hallucinations and ensuring factual responses based on the document.

## How It Works

The application follows a standard RAG pipeline:

1.  **Indexing:**
    - The source text document (`club_data.txt` or an uploaded file) is read and split into smaller, overlapping chunks.
    - Each chunk is converted into a numerical vector (embedding) using an embedding model (e.g., `openai/text-embedding-3-small`).
    - These vectors are stored in a simple, in-memory `SimpleVectorStore` which uses `sklearn.neighbors.NearestNeighbors` for efficient searching.

2.  **Retrieval & Re-ranking:**
    - When a user submits a query, it's also converted into an embedding.
    - The vector store performs a cosine similarity search to find the `TOP_K` most relevant text chunks from the document.
    - These initial results are then re-ranked. A `combined_score` is calculated for each chunk, which is a weighted sum of the initial similarity score and a keyword overlap score (measuring how many query words appear in the chunk). This prioritizes chunks that are both semantically related and contain exact keywords.

3.  **Generation:**
    - The top-ranked, re-ordered chunks are concatenated to form a `CONTEXT`.
    - This context, along with the original user query, is passed to a chat model (e.g., `openai/gpt-4o-mini`) with a strict system prompt.
    - The prompt instructs the model to formulate an answer using *only* the provided context and to state if the answer cannot be found.
    - The final answer is then displayed to the user in the Streamlit interface.

## Setup and Usage

### Prerequisites

- Python 3.8+
- An API key from [OpenRouter.ai](https://openrouter.ai/) (or an OpenAI-compatible provider).

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/kartikaysrivastava23/Club-RAG-Agent.git
    cd Club-RAG-Agent
    ```

2.  **Create a virtual environment and activate it:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install the required packages:**
    The application requires the following Python libraries. You can install them directly:
    ```bash
    pip install streamlit openai numpy scikit-learn
    ```

4.  **Set up environment variables:**
    You must set your API key as an environment variable. Create a file named `.env` in the root directory and add the following line:

    ```
    OPENROUTER_API_KEY="your-api-key-here"
    ```
    Alternatively, you can export it in your shell session:
    ```bash
    export OPENROUTER_API_KEY="your-api-key-here"
    ```

### Running the Application

1.  **Launch the Streamlit app:**
    ```bash
    streamlit run app.py
    ```

2.  **Use the agent:**
    - The application will automatically build an index from the default `club_data.txt` on the first run.
    - Type your question into the text input field and press "Ask".
    - You can use the sidebar to upload your own `.txt` file and click "Build / Rebuild Index" to use it as the new knowledge source.

## Configuration

The application can be configured using the following environment variables:

| Variable                | Description                                                                 | Default Value                    |
| ----------------------- | --------------------------------------------------------------------------- | -------------------------------- |
| `OPENROUTER_API_KEY`    | **(Required)** Your API key for OpenRouter.                                 | `None`                           |
| `OPENROUTER_API_BASE`   | The base URL for the API endpoint.                                          | `https://openrouter.ai/api/v1`   |
| `CHAT_MODEL`            | The chat model to use for generating answers.                             | `openai/gpt-4o-mini`             |
| `EMBEDDING_MODEL`       | The embedding model to use for vectorization.                             | `openai/text-embedding-3-small`  |
| `TOP_K`                 | The number of initial document chunks to retrieve for context.              | `6`                              |

## File Descriptions

-   `app.py`: The main Streamlit application file. It handles the UI, state management, and orchestrates the RAG pipeline.
-   `embeddings_utils.py`: Contains utility functions for text processing (chunking, cleaning), creating embeddings, the `SimpleVectorStore` class, and the re-ranking logic.
-   `club_data.txt`: The default knowledge base document. It contains information about the Google Developer Group on Campus (GDGC) and its events.
