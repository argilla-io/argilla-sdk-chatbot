# argilla-sdk-chatbot
Building a chatbot for Argilla SDK Step by step.

This repository contains all the scripts and notebooks that were used to create a ChatBot to help with the new Argilla 2.0 SDK.

The blog is at `argilla-chatbot.md` file. Will share the link to the deployed version once accepted at https://github.com/huggingface/blog.

The blogpost gives a tour of the different components needed to create a chatbot (in this case for the Argilla 2.0 SDK, but can be applied to any other repository) just using the raw documentation in a repository, and interact with the final app deployed at a Hugging Face Space.

To interact with the chatbot, you can visit the following link:

> https://huggingface.co/spaces/plaguss/argilla-sdk-chatbot-space

## Contents

Each section contains a brief explanation of what the script/notebook does, how to run it and what artifacts should be obtained from it.

### 1. `docs_dataset.py`

The first script to run. It goes through a repo docs folder (must be in markdown), creates chunks from it and creates a dataset.

Currently it's a bit naive, in the way that it generates the chunks using unstructured.io's [`chunk_by_title`](https://docs.unstructured.io/api-reference/api-services/chunking#by-title-chunking-strategy) (with the default values from the function, nothing else was tweaked).

It's a really short script to showcase the first step.

Needs the requirements from `requirements/requirements_docs_dataset.txt`.

Run:

```bash
python docs_dataset.py \
    "argilla-io/argilla-python" \
    --dataset-name "plaguss/argilla_sdk_docs_raw_unstructured"
```

It will generate the following dataset: [plaguss/argilla_sdk_docs_raw_unstructured](https://huggingface.co/datasets/plaguss/argilla_sdk_docs_raw_unstructured).


### 2. `pipeline_docs_queries.py`

Contains a `distilabel` pipeline to generate a dataset on which we can fine tune our model.

Needs [`distilabel`](https://github.com/argilla-io/distilabel). Uses the Hugging Face's Inference Endpoints.

Run it as:

```bash
python docs_dataset.py \
    "argilla-io/argilla-python" \
    --dataset-name "plaguss/argilla_sdk_docs_raw_unstructured"
```

Will generate a dataset like the following: [plaguss/argilla_sdk_docs_queries](https://huggingface.co/datasets/plaguss/argilla_sdk_docs_queries).

### 3. `argilla_datasets.ipynb`

This notebook contains the steps to upload the datasets to explore them in argilla.

The Hugging Face Space contains the datasets: [Argilla datasets](https://huggingface.co/spaces/plaguss/argilla-sdk-chatbot). They use the same names as in the Hugging Face Hub.

### 4. `train_embeddings.ipynb`

It follows [this blog](https://www.philschmid.de/fine-tune-embedding-model-for-rag) from Phillip Schmid to fine tune a model on the dataset we previously generated.

Needs the requirements from `requirements/requirements_training.txt`.

Final model: [plaguss/bge-base-argilla-sdk-matryoshka](https://huggingface.co/plaguss/bge-base-argilla-sdk-matryoshka).

The model doesn't improve that much as can be seen in the notebook, but does the job.

### 5. `vector_db.ipynb`

Creates the vector database. Uses [lancedb](https://lancedb.github.io/lancedb/), it's a small embedded database that can be easily moved.

The database is a file stored with the [distiset](https://huggingface.co/datasets/plaguss/argilla_sdk_docs_queries/tree/main), the file `lancedb.tar.gz`. This file can be moved as any other artifact, for example download it in the same place the app will be deployed.

### 6. `app/`

Contains the gradio app with the chatbot.

It's a basich Chat app with gradio. Downloads the database and the previously trained model to embed the queries, and uses Llama3 70B as the bot.

The prompt is **heavily** inspired from that of [wandbot](https://github.com/wandb/wandbot/blob/main/src/wandbot/rag/response_synthesis.py).

To run it locally.

```bash
python app/app.py
```

Needs more guardrails, but some initial examples can be seen already:

(still need to remove the *eot_id*)

- One of the queries from the examples:

![alt text](/assets/img_1.png)

- An unrelated question:

![alt text](assets/img_2.png)

- Ask it to ignore the system prompt:

![alt text](assets/img_3.png)

## TODO

- [ ] *The dataset should be rebuilt with an updated version of the docs?: [updated docs](https://github.com/argilla-io/argilla/tree/develop/argilla)*

- [ ] Update code to use `client.chat_completion`, for reference: https://github.com/argilla-io/distilabel/issues/772.

## Next steps

- [ ] Include references/sources for the responses, so we can give some URL to visit.
