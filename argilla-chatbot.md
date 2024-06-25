---
title: "Creating a chatbot for the new Argilla SDK: leveraging distilabel to fine tune a custom Embedding model for RAG"
thumbnail: /blog/assets/101_decision-transformers-train/thumbnail.gif
authors:
- user: plaguss
---

# Creating a chatbot for the new Argilla SDK: leveraging distilabel to fine tune a custom Embedding model for RAG

## TODO

*The dataset was built using an outdated version of the documentation, we should rebuild it pointing to the updated ones: [updated docs](https://github.com/argilla-io/argilla/tree/develop/argilla)*

## TL;DR

ADD SUMMARY HERE.

## Table of Contents

- [ADD MOTIVATION HERE](#motivation)
- [Generating Synthetic Data for Fine-Tuning Custom Embedding Models](#generating-synthetic-data-for-fine-tuning-custom-embedding-models)
    - [Downloading and chunking data](#downloading-and-chunking-data)
    - [Generating synthetic data for our embedding model using distilabel](#generating-synthetic-data-for–our-embedding-model-using-distilabel)
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

ADD INTRODUCTION OF THE STEPS, AND WHY THE TYPE OF PIPELINE DEFINED. WANTED TO TRY THE TRIPLETS, OTHER STRATEGIES COULD ALSO WORK.

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

### Generating synthetic data for our embedding model using distilabel

This section roughly cover steps 3 and 4 of the previous [X post](https://x.com/_philschmid/status/1798388387822317933), we will generate synthetic questions from our documentation that can be answered by every chunk of documentation, and hard negatives to form triplets for the fine tuning dataset.

The full pipeline script can be seen at [`pipeline_docs_queries.py`](https://github.com/argilla-io/argilla-sdk-chatbot/blob/develop/pipeline_docs_queries.py), in the reference repository, but let's go over the different steps:

1) `load_data`:

Start by downloading the dataset that contains the documentation chunks. In this case as the column with the different chunks was named `chunks`, which is the one we want to use as `anchor`. For that, we will use the `output_mappings` to change the name of the column `chunks -> anchor` so the next step contains the expected input:

```python
load_data = LoadDataFromHub(
    name="load_data",
    repo_id="plaguss/argilla_sdk_docs_raw_unstructured",
    output_mappings={"chunks": "anchor"},
    batch_size=10,
)
```

2) `generate_sentence_pair`

The next step is the most interesting, we will use [`GenerateSentencePair`](https://distilabel.argilla.io/latest/components-gallery/tasks/generatesentencepair/) to generate synthetic queries from our documentation pieces. You can take a look at the documentation to see the possibilities this `Task` opens for us, but let's go over the case at hands:

```python
llm = InferenceEndpointsLLM(
    model_id="meta-llama/Meta-Llama-3-70B-Instruct",
    tokenizer_id="meta-llama/Meta-Llama-3-70B-Instruct",
)

generate_sentence_pair = GenerateSentencePair(
    name="generate_sentence_pair",
    triplet=True,  # Generate positive and negative
    action="query",
    context="The generated sentence has to be related with Argilla, a data annotation tool for AI engineers and domain experts.",
    llm=llm,
    input_batch_size=10,
    output_mappings={"model_name": "model_name_query"},
)
```

This task will generate a series of triplets (`triplet=True`), that is, a set of (`anchor-positive-negative`) rows, a format that can be directly used for fine tuning (take a look at [training overview](https://www.sbert.net/docs/sentence_transformer/training_overview.html) in Sentence Transformers library for more information on the available formats). By using `action="query"` we request the `LLM` that generates queries for the `positive` sentences. The `context` argument helps the model when the given `anchor` doesn't contain a lot of information on itself, as it can be the case with some of the chunks of documentation we obtained. For the `LLM` we choose to use `meta-llama/Meta-Llama-3-70B-Instruct` via [`InferenceEndpointsLLM`](https://distilabel.argilla.io/latest/components-gallery/llms/inferenceendpointsllm/).

3) `multiply_queries`

Using the `GenerateSentencePair` step we obtained as much examples for training as chunks we had, 251 in this case, which can be a small number to fine tune a custom model. In order to have a bigger dataset, we can leverage another `LLM` to generate more queries. This is the target of the following `Task`, the only custom task we had to define for this pipeline:

```python
multiply_queries = MultipleQueries(
    name="multiply_queries",
    num_queries=3,
    system_prompt=(
        "You are an AI assistant helping to generate diverse examples. Ensure the "
        "generated queries are all in separated lines and preceded by a dash. "
        "Do not generate anything else or introduce the task."
    ),
    llm=llm,
    input_batch_size=10,
    input_mappings={"query": "positive"},
    output_mappings={"model_name": "model_name_query_multiplied"},
)
```

The most relevant parameter is `num_queries`, here set to 3 (which means we will generate 3 extra "positive" queries), so se would have approximately (take into account some examples can fail) *4x* the number of examples. Other than that, we use the `system_prompt` to guide the `LLM` to adhere to the our instructions. The model chosen is strong, and the examples were simple enough, so we didn't need to use structured generation, but this can be a good use case.

In case you want to see the definition of the full `Task`, open the following dropdown to see how it's defined:

<details close>
<summary>MultipleQueries definition</summary>
<br>

```python
multiply_queries_template = (
    "Given the following query:\n{original}\nGenerate {num_queries} similar queries by varying "
    "the tone and the phrases slightly. "
    "Ensure the generated queries are coherent with the original reference and relevant to the context of data annotation "
    "and AI dataset development."
)

class MultipleQueries(Task):
    system_prompt: Optional[str] = None
    num_queries: int = 1

    @property
    def inputs(self) -> List[str]:
        return ["query"]

    def format_input(self, input: Dict[str, Any]) -> ChatType:
        prompt = [
            {
                "role": "user",
                "content": multiply_queries_template.format(
                    original=input["query"],
                    num_queries=self.num_queries
                ),
            },
        ]
        if self.system_prompt:
            prompt.insert(0, {"role": "system", "content": self.system_prompt})
        return prompt

    @property
    def outputs(self) -> List[str]:
        return ["queries", "model_name"]

    def format_output(
        self, output: Union[str, None], input: Dict[str, Any]
    ) -> Dict[str, Any]:
        queries = output.split("- ")
        if len(queries) > self.num_queries:
            queries = queries[1:]
        queries = [q.strip() for q in queries]
        return {"queries": queries}
```

</details>

4) `combiner`

The last 2 steps of the pipeline are for processing the data. We want our final dataset to have rows of triplets to be used for fine tuning, but after generating 
multiple queries, we have two different columns in our dataset: `positive` and `queries`, that contain the original query (a `str`) and the multiple extra queries (a list of `str`) respectively, that pertain to the same entity. In order to combine them in a single list with all the queries, we will make use of the following step, [`CombineKeys`](https://distilabel.argilla.io/dev/components-gallery/steps/combinekeys/):

```python
combiner = CombineKeys(
    name="combiner",
    keys=["positive", "queries"],
    output_key="positive"
)
```

5) `expand_columns`

And lastly we make use of [`ExpandColumns`](https://distilabel.argilla.io/dev/components-gallery/steps/expandcolumns/) to move the previous column os positive to different lines, repeating the `anchor` and `negative` columns in the process:

```python
expand_columns = ExpandColumns(columns=["positive"])
```

With this last step, we obtain the final [`distiset`](https://huggingface.co/datasets/plaguss/argilla_sdk_docs_queries), that contains close to 1K rows with triplets of `anchor` (the original chunks), `positive` (the synthetic queries, 4 in total per anchor) and `negative` (the irrelevant query, only 1 per anchor, but repeated per query).

Click the dropdown to see the full pipeline definition:

<details close>
<summary>Distilabel Pipeline</summary>
<br>

```python
from pathlib import Path
from typing import Any, Dict, List, Union, Optional

from distilabel.pipeline import Pipeline
from distilabel.steps import LoadDataFromHub
from distilabel.llms import InferenceEndpointsLLM
from distilabel.steps.tasks import GenerateSentencePair
from distilabel.steps.tasks.base import Task
from distilabel.steps.tasks.typing import ChatType
from distilabel.steps import ExpandColumns, CombineKeys


multiply_queries_template = (
    "Given the following query:\n{original}\nGenerate {num_queries} similar queries by varying "
    "the tone and the phrases slightly. "
    "Ensure the generated queries are coherent with the original reference and relevant to the context of data annotation "
    "and AI dataset development."
)

class MultipleQueries(Task):
    system_prompt: Optional[str] = None
    num_queries: int = 1

    @property
    def inputs(self) -> List[str]:
        return ["query"]

    def format_input(self, input: Dict[str, Any]) -> ChatType:
        prompt = [
            {
                "role": "user",
                "content": multiply_queries_template.format(
                    original=input["query"],
                    num_queries=self.num_queries
                ),
            },
        ]
        if self.system_prompt:
            prompt.insert(0, {"role": "system", "content": self.system_prompt})
        return prompt

    @property
    def outputs(self) -> List[str]:
        return ["queries", "model_name"]

    def format_output(
        self, output: Union[str, None], input: Dict[str, Any]
    ) -> Dict[str, Any]:
        queries = output.split("- ")
        if len(queries) > self.num_queries:
            queries = queries[1:]
        queries = [q.strip() for q in queries]
        return {"queries": queries}


with Pipeline(
    name="embedding-queries",
    description="Generate queries to train a sentence embedding model."
) as pipeline:
    load_data = LoadDataFromHub(
        name="load_data",
        repo_id="plaguss/argilla_sdk_docs_raw_unstructured",
        output_mappings={"chunks": "anchor"},
        batch_size=10,
    )

    llm = InferenceEndpointsLLM(
        model_id="meta-llama/Meta-Llama-3-70B-Instruct",
        tokenizer_id="meta-llama/Meta-Llama-3-70B-Instruct",
    )

    generate_sentence_pair = GenerateSentencePair(
        name="generate_sentence_pair",
        triplet=True,  # Generate positive and negative
        action="query",
        context="The generated sentence has to be related with Argilla, a data annotation tool for AI engineers and domain experts.",
        llm=llm,
        input_batch_size=10,
        output_mappings={"model_name": "model_name_query"},
    )

    multiply_queries = MultipleQueries(
        name="multiply_queries",
        num_queries=3,
        system_prompt=(
            "You are an AI assistant helping to generate diverse examples. Ensure the "
            "generated queries are all in separated lines and preceded by a dash. "
            "Do not generate anything else or introduce the task."
        ),
        llm=llm,
        input_batch_size=10,
        input_mappings={"query": "positive"},
        output_mappings={"model_name": "model_name_query_multiplied"},
    )

    combiner = CombineKeys(
        name="combiner",
        keys=["positive", "queries"],
        output_key="positive"
    )

    expand_columns = ExpandColumns(
        columns=["positive"],
    )

    (
        load_data
        >> generate_sentence_pair
        >> multiply_queries
        >> combiner
        >> expand_columns
    )


if __name__ == "__main__":

    pipeline_parameters = {
        "generate_sentence_pair": {
            "llm": {
                "generation_kwargs": {
                    "temperature": 0.7,
                    "max_new_tokens": 512,
                }
            }
        },
        "multiply_queries": {
            "llm": {
                "generation_kwargs": {
                    "temperature": 0.7,
                    "max_new_tokens": 512,
                }
            }
        }
    }

    distiset = pipeline.run(
        parameters=pipeline_parameters
    )
    distiset.push_to_hub("plaguss/argilla_sdk_docs_queries")
```

</details>

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
- Implement some deduplication/filter pairs in the training dataset.

## References

- [X on Creating a Pipeline for Generating Synthetic Data for Fine-Tuning Custom Embedding Models.](https://x.com/_philschmid/status/1798388387822317933)
- [fine-tune-embedding-model-for-rag](https://www.philschmid.de/fine-tune-embedding-model-for-rag)
- [wandbot](https://github.com/wandb/wandbot/tree/main)
