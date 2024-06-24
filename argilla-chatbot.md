---
title: "Creating a chatbot for the new Argilla SDK: leveraging distilabel to fine tune a custom Embedding model for RAG"
thumbnail: /blog/assets/101_decision-transformers-train/thumbnail.gif
authors:
- user: plaguss
---

# Creating a chatbot for the new Argilla SDK: leveraging distilabel to fine tune a custom Embedding model for RAG

## TODO

*The dataset was built using an outdated version of the documentation, we should rebuild it pointing to the updated ones: [updated docs](https://github.com/argilla-io/argilla/tree/develop/argilla)*

## TL;DR

ADD SUMMARY HERE.

## Table of Contents

- [ADD MOTIVATION HERE](#motivation)
- [Generating Synthetic Data for Fine-Tuning Custom Embedding Models](#generating-synthetic-data-for-fine-tuning-custom-embedding-models)
    - [Downloading and chunking data](#downloading-and-chunking-data)
    - [Generating synthetic data for our embedding model: Distilabel to the rescue](#generating-synthetic-data-for–our-embedding-model:-distilabel-to-the-rescue)
    - [Explore the datasets in Argilla](#explore-the-datasets-in-argilla)
    - [Fine-Tune the embedding model](#fine-tune-the-embedding-model)
- [The vector database](#create-the-vector-database)
- [Creating our ChatBot](#next-steps)
    - [Creating a Gradio App](#creating-the-gradio-app)
    - [Deploy the ChatBot app on Hugging Face Spaces](#deploy-the-chatbot-app-on-hugging-face-spaces)
    - [Playing around with our chatbot](#play-around-with-your-chatbot)    
- [Next Steps](#next-steps)

## MOTIVATION

TBD

Take a look at the following repository to see all the code that led to this blogpost at [argilla-sdk-chatbot](https://github.com/argilla-io/argilla-sdk-chatbot),
we will refer to scripts or notebooks that can be found there.

## Generating Synthetic Data for Fine-Tuning Custom Embedding Models

> TODO: Mention here the X message and blogpost shown in the references.

### Downloading and chunking data

Chunk the Data: Divide your text data into manageable chunks of approximately 256 tokens each (chunk size used in RAG later).

The first step consists on processing the documentation of your target repository. There are some libraries you can use out of the box to read the contents of a repository (like [llama-index](https://docs.llamaindex.ai/en/stable/examples/data_connectors/GithubRepositoryReaderDemo/)) and parse the markdown content (langchain has a [MarkdownTextSplitter](https://python.langchain.com/v0.1/docs/modules/data_connection/document_transformers/markdown_header_metadata/), and there's the [MarkdownNodeParser](https://docs.llamaindex.ai/en/stable/module_guides/loading/node_parsers/modules/?h=markdown#markdownnodeparser) from `llama-index`), or if you just want everything abstracted in a interface, try this cool [corpus-creator](https://huggingface.co/spaces/davanstrien/corpus-creator) app from [`davanstrien`](https://huggingface.co/davanstrien).

We want abstract the process a bit less, yet keep it simple, so we created a custom python script, that can be found in the repo [here](https://github.com/argilla-io/argilla-sdk-chatbot/blob/develop/docs_dataset.py). Let's see how to use it:


```console
python docs_dataset.py \
    "argilla-io/argilla-python" \
    --dataset-name "plaguss/argilla_sdk_docs_raw_unstructured"
```

There are some extra arguments that we can pass, but the minimum required are the path to the repository where the docs can be found, and the dataset that we want for the dataset in the Hugging Face Hub.

The script will download the docs (placed at `/docs` by default, but this can be changed as we can see in the following snippet) to your local repository, extract all the markdown files it finds there, chunk them, and push the dataset to the Hugging Face Hub. The core logic can be summarized by the following snippet:

```python
gh = Github()
repo = gh.get_repo("repo_name")

# Download the 
download_folder(repo, "/folder/with/docs", "dir/to/download/docs")

# Extract the markdown files from the downloaded folder with the documentation from the GitHub repository
md_files = list(docs_path.glob("**/*.md"))

# Loop to iterate over the files and generate chunks from the text pieces
data = create_chunks(md_files)

# Create a dataset to push it to the hub
create_dataset(data, repo_name="name/of/the/dataset")
```

The script contains some short functions to download the code, create the chunks out of the markdown files, and create the dataset. It can be easily updated to include more functionality, or make a more complex chunking strategy, it should be easy to explore.

Take a look at the remaining arguments that can be tweaked by calling the help message:

```console
python docs_dataset.py -h
```

### Generating synthetic data for our embedding model: Distilabel to the rescue

GO OVER THE SCRIPT WITH THE DISTILABEL PIPELINE.

### Explore the datasets in Argilla

It's moment to make use of Argilla to explore the datasets we have generated and iterate on them as needed. You can follow the [argilla_datasets.ipynb](https://github.com/argilla-io/argilla-sdk-chatbot/blob/develop/argilla_datasets.ipynb) notebook to see how to upload the datasets to Argilla.

If you don't have an Argilla instance running, you can follow the guide from the [docs](https://argilla-io.github.io/argilla/dev/getting_started/quickstart/#run-the-argilla-server), and just create a Hugging Face space with Argilla. Once you have the Space up and running, you can easily connect to it (update the `api_url` to point to your space):

```python
import argilla as rg

client = rg.Argilla(
    api_url="https://plaguss-argilla-sdk-chatbot.hf.space",
    api_key="owner.apikey"
)
```

#### An Argilla dataset with chunks of technical documentation

Once we are connected, we will create the `Settings` for our dataset. These `Settings` should work for your use case without changes:

```python
settings = rg.Settings(
    guidelines="Review the chunks of docs.",
    fields=[
        rg.TextField(
            name="filename",
            title="Filename where this chunk was extracted from",
            use_markdown=False,
        ),
        rg.TextField(
            name="chunk",
            title="Chunk from the documentation",
            use_markdown=False,
        ),
    ],
    questions=[
        rg.LabelQuestion(
            name="good_chunk",
            title="Does this chunk contain relevant information?",
            labels=["yes", "no"],
        )
    ],
)
```

We will explore the filename that was parsed, and the chunks generated by our program, hence the `filename` and `chunk` fields. We can also define a simple label question (`good_chunk`) to label the chunks as useful or not and improve on the automated generation, and that would be it, we can push the configuration:

```python
dataset = rg.Dataset(
    name="argilla_sdk_docs_raw_unstructured",
    settings=settings,
    client=client,
)
dataset.create()
```

Let's grab the dataset from the Hugging Face hub. Download the dataset created in section `Downloading and chunking data`, which in our case corresponds to the following dataset, and filter the columns we need:

```python
from datasets import load_dataset

data = (
    load_dataset("plaguss/argilla_sdk_docs_raw_unstructured", split="train")
    .select_columns(["filename", "chunks"])
    .to_list()
)
```

And we are ready for the last step, log the records to see them in the argilla screen:

```python
dataset.records.log(records=data, mapping={"filename": "filename", "chunks": "chunk"})
```

These are the kind of examples you could expect to see:

![argilla-img-1](/assets/blog/argilla-img-1.png)

#### An Argilla dataset with triplets to fine tune an embedding model

Now we can repeat the process with the dataset ready for fine tuning we generated in the [previous section](#generating-synthetic-data-for–our-embedding-model:-distilabel-to-the-rescue). We will show just the new settings for this dataset, we only need to download the corresponding dataset and push it with it's corresponding name. The jupyter notebook contains all the details:

```python
settings = rg.Settings(
    guidelines="Review the chunks of docs.",
    fields=[
        rg.TextField(
            name="anchor",
            title="Anchor (Chunk from the documentation).",
            use_markdown=False,
        ),
        rg.TextField(
            name="positive",
            title="Positive sentence that queries the anchor.",
            use_markdown=False,
        ),
        rg.TextField(
            name="negative",
            title="Negative sentence that may use similar words but has content unrelated to the anchor.",
            use_markdown=False,
        ),
    ],
    questions=[
        rg.LabelQuestion(
            name="is_positive_relevant",
            title="Is the positive query relevant?",
            labels=["yes", "no"],
        ),
        rg.LabelQuestion(
            name="is_negative_irrelevant",
            title="Is the negative query irrelevant?",
            labels=["yes", "no"],
        )
    ],
)
```

In this case we have 3 `TextFields` as we have the `anchor`, `positive` and `negative`, which correspond to the chunk of text, a query that could be answered using the chunk as reference, and an (un)related query to work as a negative in the triplet respectively. The two questions can be used to discriminate this `positive/negative` examples.

An example can be seen in the following image:

![argilla-img-2](/assets/blog/argilla-img-2.png)

This dataset setting was made to explore the dataset, but we could use prepare it to find wrong examples, improve the questions generated, and iterate on the dataset to be used in the following section.

### Fine-Tune the embedding model

This will be a guide over the blog already done.

## The vector database

## Creating our ChatBot

### Creating a Gradio App

### Deploy the ChatBot app on Hugging Face Spaces

### Playing around with our ChatBot

## Next steps

- Improve the chunking strategy: Explore new chunk strategies, play around with different parameters, chunk sizes, etc...

## References

- [X on Creating a Pipeline for Generating Synthetic Data for Fine-Tuning Custom Embedding Models.](https://x.com/_philschmid/status/1798388387822317933)
- [fine-tune-embedding-model-for-rag](https://www.philschmid.de/fine-tune-embedding-model-for-rag)
- [wandbot](https://github.com/wandb/wandbot/tree/main)
