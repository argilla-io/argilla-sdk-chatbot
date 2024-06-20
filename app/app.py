from typing import Optional, Any
import os
from pathlib import Path
import tarfile
from dataclasses import dataclass

import lancedb
from lancedb.embeddings import get_registry
from huggingface_hub.file_download import hf_hub_download
from huggingface_hub import InferenceClient
from transformers import AutoTokenizer
import gradio as gr


def untar_file(source: Path) -> Path:
    """Untar and decompress files which have passed by `make_tarfile`.

    Args:
        source (Path): Path pointing to a .tag.gz file.

    Returns:
        filename (Path): The filename of the file decompressed.
    """
    new_filename = source.parent / source.stem.replace(".tar", "")
    with tarfile.open(source, "r:gz") as f:
        f.extractall(source.parent)
    return new_filename


@dataclass
class Settings:
    LANCEDB: str = "lancedb"
    LANCEDB_FILE_TAR: str = "lancedb.tar.gz"
    TOKEN: str = os.getenv("HF_API_TOKEN")
    LOCAL_DIR: Path = Path.home() / ".cache/argilla_sdk_docs_db"
    REPO_ID: str = "plaguss/argilla_sdk_docs_queries"
    TABLE_NAME: str = "docs"
    MODEL_NAME: str = "plaguss/bge-base-argilla-sdk-matryoshka"
    DEVICE: str = "mps"


settings = Settings()

# Get the model to create the embeddings
model = get_registry().get("sentence-transformers").create(name=settings.MODEL_NAME, device=settings.DEVICE)

# from lancedb.pydantic import LanceModel, Vector
# class Docs(LanceModel):
#     query: str = model.SourceField()
#     text: str = model.SourceField()
#     vector: Vector(model.ndims()) = model.VectorField()


class Database:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.table = self.get_table_from_db()

    def get_table_from_db(self) -> lancedb.table.LanceTable:
        lancedb_db_path = self.settings.LOCAL_DIR / self.settings.LANCEDB
        if not lancedb_db_path.exists():
            lancedb_download = Path(
                hf_hub_download(
                    self.settings.REPO_ID,
                    self.settings.LANCEDB_FILE_TAR,
                    repo_type="dataset",
                    token=self.settings.TOKEN,
                    local_dir=self.settings.LOCAL_DIR
                )
            )

            lancedb_db_path = untar_file(lancedb_download)

        db = lancedb.connect(str(lancedb_db_path))
        table_name = "docs"
        table = db.open_table(table_name)
        return table

    def retrieve_doc_chunks(self, query: str, limit: int = 12, hard_limit: int = 4) -> str:
        # TODO: THE QUERY FOR THE SEARCH MUST BE A VECTOR, SO WE HAVE TO EMBED THE DATA FIRST
        # embedded_query = model.generate_embeddings([query])
        retrieved = (
            self.table
                .search(query)
                # .search(embedded_query[0])
                .metric("cosine")
                .limit(limit)
                .select(["text"])  # Just grab the chunk to use for context
                .to_list()
        )
        # We have repeated questions (up to 4) for a given chunk, so we may get repeated chunks.
        # Request more than necessary and filter them afterwards
        responses = []
        unique_responses = set()

        for item in retrieved:
            chunk = item["text"]
            if chunk not in unique_responses:
                unique_responses.add(chunk)
                responses.append(chunk)

        context = ""
        for i, item in enumerate(responses[:hard_limit]):
            if i > 0:
                context += "\n\n"
            context += f"---\n{item}"
        return context


database = Database(settings=settings)


def get_client_and_tokenizer(
    model_id: str = "meta-llama/Meta-Llama-3-70B-Instruct",
    tokenizer_id: Optional[str] = None
) -> InferenceClient:
    if tokenizer_id is None:
        tokenizer_id = model_id

    client = InferenceClient()
    base_url = client._resolve_url(
        model=model_id, task="text-generation"
    )
    client = InferenceClient(
        model=base_url,
        token=os.getenv("HF_API_TOKEN")
    )
    # TODO: Move to an async client
    #client = AsyncInferenceClient(
    #    model=base_url,
    #    token=api_key,
    #)

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)
    return client, tokenizer


client_kwargs = {
    "stream": True,
    "max_new_tokens": 512,
    "do_sample": False,
    "typical_p": None,
    "repetition_penalty": None,
    "temperature": 0.3,
    "top_p": None,
    "top_k": None,
    "stop_sequences": None,
    "seed": None,
}


client, tokenizer = get_client_and_tokenizer()

SYSTEM_PROMPT = """\
You are a support expert in Argilla SDK, whose goal is help users with their questions.
As a trustworthy expert, you must provide truthful answers to questions using only the provided documentation snippets, not prior knowledge.
Here are guidelines you must follow when responding to user questions:

##Purpose and Functionality**
- Answer questions related to the Argilla SDK.
- Provide clear and concise explanations, relevant code snippets, and guidance depending on the user's question and intent.
- Ensure users succeed in effectively understanding and using Argilla's features.
- Provide accurate responses to the user's questions.

**Specificity**
- Be specific and provide details only when required.
- Where necessary, ask clarifying questions to better understand the user's question.
- Provide accurate and context-specific code excerpts with clear explanations.
- Ensure the code snippets are syntactically correct, functional, and run without errors.
- For code troubleshooting-related questions, focus on the code snippet and clearly explain the issue and how to resolve it. 
- Avoid boilerplate code such as imports, installs, etc.

**Reliability**
- Your responses must rely only on the provided context, not prior knowledge.
- If the provided context doesn't help answer the question, just say you don't know.
- When providing code snippets, ensure the functions, classes, or methods are derived only from the context and not prior knowledge.
- Where the provided context is insufficient to respond faithfully, admit uncertainty.
- Remind the user of your specialization in Argilla SDK support when a question is outside your domain of expertise.
- Redirect the user to the appropriate support channels - Argilla [community](https://join.slack.com/t/rubrixworkspace/shared_invite/zt-whigkyjn-a3IUJLD7gDbTZ0rKlvcJ5g) when the question is outside your capabilities or you do not have enough context to answer the question.

**Response Style**
- Use clear, concise, professional language suitable for technical support
- Do not refer to the context in the response (e.g., "As mentioned in the context...") instead, provide the information directly in the response.

**Example**:

The correct answer to the user's query

 Steps to solve the problem:
 - **Step 1**: ...
 - **Step 2**: ...
 ...

 Here's a code snippet

 ```python
 # Code example
 ...
 ```
 
 **Explanation**:

 - Point 1
 - Point 2
 ...
"""

ARGILLA_BOT_TEMPLATE = """\
Please provide an answer to the following question related to Argilla's new SDK.

You can make use of the chunks of documents in the context to help you generating the response.

## Query:
{message}

## Context:
{context}
"""


def prepare_input(message: str, history: Any):
    # Retrieve the context from the database
    context = database.retrieve_doc_chunks(message)

    # Prepare the conversation for the model.
    conversation = []
    for human, bot in history:
        conversation.append({"role": "user", "content": human})
        conversation.append({"role": "assistant", "content": bot})

    conversation.insert(0, {"role": "system", "content": SYSTEM_PROMPT})
    conversation.append(
        {
            "role": "user",
            "content": ARGILLA_BOT_TEMPLATE.format(message=message, context=context),
        }
    )

    return tokenizer.apply_chat_template(
        [conversation],
        tokenize=False,
        add_generation_prompt=True,
    )[0]


def chatty(message, history):
    prompt = prepare_input(message, history)

    partial_message = ""
    for token_stream in client.text_generation(prompt=prompt, **client_kwargs):
        partial_message += token_stream
        yield partial_message



if __name__ == "__main__":

    import gradio as gr
    gr.ChatInterface(
        chatty,
        chatbot=gr.Chatbot(height=600),
        textbox=gr.Textbox(placeholder="Ask me about the new argilla SDK", container=False, scale=7),
        title="Argilla SDK Chatbot",
        description="Ask a question about Argilla SDK",
        theme="soft",
        examples=[
            "How can I connect to an argilla server?",
            "How can I access a dataset?",
            "How can I get the current user?"
        ],
        cache_examples=True,
        retry_btn=None,
        # undo_btn="Delete Previous",
        # clear_btn="Clear",
    ).launch()
