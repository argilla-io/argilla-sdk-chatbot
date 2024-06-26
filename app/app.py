from typing import Optional, Generator
import os
from pathlib import Path
import tarfile
from dataclasses import dataclass

import torch
import lancedb
from lancedb.embeddings import get_registry
from huggingface_hub.file_download import hf_hub_download
from huggingface_hub import InferenceClient, login
from transformers import AutoTokenizer
import gradio as gr


@dataclass
class Settings:
    """Settings class to store useful variables for the App."""

    LANCEDB: str = "lancedb"
    LANCEDB_FILE_TAR: str = "lancedb.tar.gz"
    TOKEN: str = os.getenv("HF_API_TOKEN")
    LOCAL_DIR: Path = Path.home() / ".cache/argilla_sdk_docs_db"
    REPO_ID: str = "plaguss/argilla_sdk_docs_queries"
    TABLE_NAME: str = "docs"
    MODEL_NAME: str = "plaguss/bge-base-argilla-sdk-matryoshka"
    DEVICE: str = (
        "mps"
        if torch.backends.mps.is_available()
        else "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )
    MODEL_ID: str = "meta-llama/Meta-Llama-3-70B-Instruct"


settings = Settings()

login(token=settings.TOKEN)


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


def download_database(
    repo_id: str,
    lancedb_file: str = "lancedb.tar.gz",
    local_dir: Path = Path.home() / ".cache/argilla_sdk_docs_db",
    token: str = os.getenv("HF_API_TOKEN"),
) -> Path:
    """Helper function to download the database. Will download a compressed lancedb stored
    in a Hugging Face repository.

    Args:
        repo_id: Name of the repository where the databsase file is stored.
        lancedb_file: Name of the compressed file containing the lancedb database.
            Defaults to "lancedb.tar.gz".
        local_dir: Path where the file will be donwloaded to. Defaults to
            Path.home()/".cache/argilla_sdk_docs_db".
        token: Token for the Hugging Face hub API. Defaults to os.getenv("HF_API_TOKEN").

    Returns:
        The path pointing to the database already uncompressed and ready to be used.
    """
    lancedb_download = Path(
        hf_hub_download(
            repo_id, lancedb_file, repo_type="dataset", token=token, local_dir=local_dir
        )
    )
    return untar_file(lancedb_download)


# Get the model to create the embeddings
model = (
    get_registry()
    .get("sentence-transformers")
    .create(name=settings.MODEL_NAME, device=settings.DEVICE)
)


class Database:
    """Interaction with the vector database to retrieve the chunks.

    On instantiation, will donwload the lancedb database if nos already found in
    the expected location. Once ready, the only functionality available is
    to retrieve the doc chunks to be used as examples for the LLM.
    """

    def __init__(self, settings: Settings) -> None:
        """
        Args:
            settings: Instance of the settings.
        """
        self.settings = settings
        self._table: lancedb.table.LanceTable = self.get_table_from_db()

    def get_table_from_db(self) -> lancedb.table.LanceTable:
        """Downloads the database containing the embedded docs.

        If the file is not found in the expected location, will download it, and
        then create the connection, open the table and pass it.

        Returns:
            The table of the database containing the embedded chunks.
        """
        lancedb_db_path = self.settings.LOCAL_DIR / self.settings.LANCEDB

        if not lancedb_db_path.exists():
            lancedb_db_path = download_database(
                self.settings.REPO_ID,
                lancedb_file=self.settings.LANCEDB_FILE_TAR,
                local_dir=self.settings.LOCAL_DIR,
                token=self.settings.TOKEN,
            )

        db = lancedb.connect(str(lancedb_db_path))
        table = db.open_table(self.settings.TABLE_NAME)
        return table

    def retrieve_doc_chunks(
        self, query: str, limit: int = 12, hard_limit: int = 4
    ) -> str:
        """Search for similar queries in the database, and return the context to be passed
        to the LLM.

        Args:
            query: Query from the user.
            limit: Number of similar items to retrieve. Defaults to 12.
            hard_limit: Limit of responses to take into account.
                As we generated repeated questions initially, the database may contain
                repeated chunks of documents, in the initial `limit` selection, using
                `hard_limit` we limit to this number the total of unique retrieved chunks.
                Defaults to 4.

        Returns:
            The context to be used by the model to generate the response.
        """
        # Embed the query to use our custom model instead of the default one.
        embedded_query = model.generate_embeddings([query])
        field_to_retrieve = "text"
        retrieved = (
            self._table.search(embedded_query[0])
            .metric("cosine")
            .limit(limit)
            .select([field_to_retrieve])  # Just grab the chunk to use for context
            .to_list()
        )
        return self._prepare_context(retrieved, hard_limit)

    @staticmethod
    def _prepare_context(retrieved: list[dict[str, str]], hard_limit: int) -> str:
        """Prepares the examples to be used in the LLM prompt.

        Args:
            retrieved: The list of retrieved chunks.
            hard_limit: Max number of doc pieces to return.

        Returns:
            Context to be used by the LLM.
        """
        # We have repeated questions (up to 4) for a given chunk, so we may get repeated chunks.
        # Request more than necessary and filter them afterwards
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
    model_id: str = settings.MODEL_ID, tokenizer_id: Optional[str] = None
) -> tuple[InferenceClient, AutoTokenizer]:
    """Obtains the inference client and the tokenizer corresponding to the model.

    Args:
        model_id: The name of the model. Currently it must be one in the free tier.
            Defaults to "meta-llama/Meta-Llama-3-70B-Instruct".
        tokenizer_id: The name of the corresponding tokenizer. Defaults to None,
            in which case it will use the same as the `model_id`.

    Returns:
        The client and tokenizer chosen.
    """
    if tokenizer_id is None:
        tokenizer_id = model_id

    client = InferenceClient()
    base_url = client._resolve_url(model=model_id, task="text-generation")
    # Note: We could move to the AsyncClient
    client = InferenceClient(model=base_url, token=os.getenv("HF_API_TOKEN"))

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
    "stop_sequences": ["<|eot_id|>", "<|end_of_text|>"]
    if settings.MODEL_ID.startswith("meta-llama/Meta-Llama-3")
    else None,
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


def prepare_input(message: str, history: list[tuple[str, str]]) -> str:
    """Prepares the input to be passed to the LLM.

    Args:
        message: Message from the user, the query.
        history: Previous list of messages from the user and the answers, as a list
            of tuples with user/assistant messages.

    Returns:
        The string with the template formatted to be sent to the LLM.
    """
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


def chatty(message: str, history: list[tuple[str, str]]) -> Generator[str, None, None]:
    """Main function of the app, contains the interaction with the LLM.

    Args:
        message: Message from the user, the query.
        history: Previous list of messages from the user and the answers, as a list
            of tuples with user/assistant messages.

    Yields:
        The streaming response, it's printed in the interface as it's being received.
    """
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
        textbox=gr.Textbox(
            placeholder="Ask me about the new argilla SDK", container=False, scale=7
        ),
        title="Argilla SDK Chatbot",
        description="Ask a question about Argilla SDK",
        theme="soft",
        examples=[
            "How can I connect to an argilla server?",
            "How can I access a dataset?",
            "How can I get the current user?",
        ],
        cache_examples=True,
        retry_btn=None,
    ).launch()
