from pathlib import Path
from typing import Any, Dict, List, Union, Optional

from distilabel.pipeline import Pipeline
from distilabel.steps import LoadDataFromHub
from distilabel.llms import InferenceEndpointsLLM
from distilabel.steps.tasks import GenerateSentencePair
from distilabel.steps.tasks.base import Task
from distilabel.steps.tasks.typing import ChatType
from distilabel.steps import ExpandColumns, CombineKeys
from distilabel.steps import ExpandColumns, MergeColumns


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

    merge_columns = MergeColumns(
        name="merge_columns",
        columns=["positive", "queries"],
        output_column="positive"
    )

    expand_columns = ExpandColumns(
        columns=["positive"],
    )

    (
        load_data
        >> generate_sentence_pair
        >> multiply_queries
        >> merge_columns
        >> expand_columns
    )


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d",
        "--dry-run",
        action=argparse.BooleanOptionalAction,
        help="Do a dry run for testing purposes.",
    )
    parser.add_argument(
        "-n",
        "--dataset-hub-name",
        help="Name of the repository dataset in the Hugging Face Hub, or folder to save the distiset locally.",
        default="plaguss/argilla_sdk_docs_queries"
    )
    args = parser.parse_args()

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

    if args.dry_run:
        distiset = pipeline.dry_run(
            batch_size=1,
            parameters=pipeline_parameters
        )
        distiset.save_to_disk(Path.home() / "Downloads" / args.dataset_hub_name)
    
    else:
        distiset = pipeline.run(
            parameters=pipeline_parameters
        )
        distiset.push_to_hub(args.dataset_hub_name)
