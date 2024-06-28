# argilla-sdk-chatbot
Building a chatbot for Argilla SDK Step by step 

> :warning: This is still work in progress

The idea is to document the process and generate a blog post from it, as well as deploying the gradio app to Hugging Face Spaces.
For the blog we should center around generating the initial dataset to fine tune the embedding model.

The idea is to open a PR once we have it ready at: https://github.com/huggingface/blog.

The blog is at `argilla-chatbot.md` file. If you think you can add/modify anything, please open a PR and add your name in the *authors*. You can take a look on how the blog works [here](https://github.com/huggingface/blog?tab=readme-ov-file#the-hugging-face-blog-repository-).

The app is deployed here:

> https://huggingface.co/spaces/plaguss/argilla-sdk-chatbot-space

## TODO

- [ ] *The dataset should be rebuilt with an updated version of the docs?: [updated docs](https://github.com/argilla-io/argilla/tree/develop/argilla)*

- [x] Create an argilla instance to store the responses from the chatbot so we can review them later.

- [ ] Include references/sources for the responses, so we can give some URL to visit.

## Contents

### `docs_dataset.py`

The first script to run, it goes through a repo docs folder (must be in markdown), creates chunks from it and creates a dataset.

Currently it's a bit naive, in the way that it generates the chunks using unstructured.io's [`chunk_by_title`](https://docs.unstructured.io/api-reference/api-services/chunking#by-title-chunking-strategy) with the default values.

It's a really short script to showcase the first step.

Needs the requirements from `requirements/requirements_docs_dataset.txt`.

Run:

```console
python docs_dataset.py \
    "argilla-io/argilla-python" \
    --dataset-name "plaguss/argilla_sdk_docs_raw_unstructured"
```

Final dataset: [plaguss/argilla_sdk_docs_raw_unstructured](https://huggingface.co/datasets/plaguss/argilla_sdk_docs_raw_unstructured).


### `pipeline_docs_queries.py`

Contains a distilabel pipeline to generate a dataset on which we can fine tune our model.

Needs distilabel. Uses the inference endpoints with the free endpoints, we should change this and inform accordingly. It works because for the moment the dataset with the chunks is small enough.

Run:

```console
python docs_dataset.py \
    "argilla-io/argilla-python" \
    --dataset-name "plaguss/argilla_sdk_docs_raw_unstructured"
```

Final dataset: [plaguss/argilla_sdk_docs_queries](https://huggingface.co/datasets/plaguss/argilla_sdk_docs_queries).

### `argilla_datasets.ipynb`

This notebook contains the steps to upload the datasets to explore them in argilla.

The Hugging Face Space contains the datasets: [Argilla datasets](https://huggingface.co/spaces/plaguss/argilla-sdk-chatbot). They use the same names as in the Hugging Face Hub.

### `train_embeddings.ipynb`

It follows the blog from Phillip Schmid to fine tune a model on the dataset we previously generated.

Needs the requirements from `requirements/requirements_training.txt`.

Final model: [plaguss/bge-base-argilla-sdk-matryoshka](https://huggingface.co/plaguss/bge-base-argilla-sdk-matryoshka).

The model doesn't improve that much as can be seen in the notebook...

### `vector_db.ipynb`

Creates the vector database. Uses [lancedb](https://lancedb.github.io/lancedb/), it's a small embedded database that can be moved as wanted.

The database is a file stored with the [distiset](https://huggingface.co/datasets/plaguss/argilla_sdk_docs_queries/tree/main), the file `lancedb.tar.gz`.

TODO:
- [ ] Make a script to create the database and push it to huggingface datasets.

### `app/`

Contains the gradio app with the chatbot.

It's a basich Chat app with gradio. Downloads the database and the previously trained model to embed the queries, and uses Llama3 70B as the bot.

> Note:
    There's no much structure, it's all placed in a single script. It would be better to make it more modular, but the idea was to iterate fast initially.

The prompt is **heavily** inspired from that of [wandbot](https://github.com/wandb/wandbot/blob/main/src/wandbot/rag/response_synthesis.py).

To run it locally.

```console
python app/app.py
```

Needs more guardrails, but some initial examples can be seen already:

(still need to remove the *eot_id*)

- One of the queries from the examples:

![alt text](/assets/img_1.png)

- An unrelated queestion:

![alt text](assets/img_2.png)

- Ask it to ignore the system prompt:

![alt text](assets/img_3.png)
