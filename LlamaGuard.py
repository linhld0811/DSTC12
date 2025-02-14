import os
import json
import argparse
import math

from tqdm import tqdm

from vllm import LLM, SamplingParams
from datasets import load_dataset

from sklearn.metrics import roc_auc_score


def main(args):
    if args.input_file is None:
        raise ValueError("input file is required")
    if args.output is None:
        out_fp = f"completed_batches/{args.input_file}_processed.jsonl"
        if not os.path.exists(os.path.dirname(out_fp)):
            os.makedirs(os.path.dirname(out_fp))
    else:
        out_fp = args.output

    dataset = load_dataset(args.input_file, data_files="test-00000-of-00001.parquet")
    prompts_to_process = []
    labels = []
    preds = []

    llm = LLM(
        model="Llama-Guard-3-1B",
        tensor_parallel_size=args.tensor_parallel_size,
        pipeline_parallel_size=args.pipeline_parallel_size,
        max_model_len=2048 * 2,
        trust_remote_code=True,
        gpu_memory_utilization=args.util,
    )
    tokenizer = llm.get_tokenizer()
    sampling_params = SamplingParams(temperature=0, max_tokens=48, logprobs=5)

    UNSAFE_TOKEN_ID = tokenizer.convert_tokens_to_ids("unsafe")

    for example in tqdm(dataset["train"]):
        chat = example["context"] + [example["response"]]
        labels.append(example["safety_label"])
        formatted_chat = []
        for i, message in enumerate(chat):
            role = "user" if i % 2 == 0 else "assistant"
            formatted_chat.append(
                {"role": role, "content": [{"type": "text", "text": message}]}
            )

        prompts_to_process.append(
            tokenizer.apply_chat_template(
                formatted_chat,
                tokenize=False,
            )
            + "\n\n"
        )

    outputs = llm.generate(prompts_to_process, sampling_params)

    with open(out_fp, "w") as file_out:
        for output in outputs:
            pred = math.exp(output.outputs[0].logprobs[0][UNSAFE_TOKEN_ID].logprob)
            preds.append(pred)
            obj = {
                "id": output.request_id,
                "response": output.outputs[0].text,
                "score": pred,
            }
            file_out.write(json.dumps(obj) + "\n")
    print(f"Saved to {out_fp}")

    auc = roc_auc_score(labels, preds)
    print(f"AUC: {auc}")

    with open("baseline_results.txt", "a") as f:
        f.write(f"{args.input_file} AUC: {auc}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="generate predictions using vllm")

    parser.add_argument("--input_file", type=str, required=False, help="input file")
    parser.add_argument("--output", type=str, help="Path to output file")
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--pipeline_parallel_size", type=int, default=1)
    parser.add_argument("--util", type=float, default=0.9)

    args = parser.parse_args()
    main(args)
