"""Script that implements Sparsified testing of Jamba in the LM-Eval Harness."""

import pickle
from argparse import ArgumentParser
from pathlib import Path

from lm_eval import simple_evaluate
from lm_eval.models.huggingface import HFLM

from manipulatte import BackEnd, load_model


parser = ArgumentParser()
parser.add_argument("task", help="Task to run")
parser.add_argument("shots", type=int, help="num_fewshot to use")
parser.add_argument(
    "--k", type=int, nargs="*", help="k for sparsification to use, can be list", default=[3]
)

if __name__ == "__main__":
    args = parser.parse_args()
    backend = BackEnd()
    model = load_model("huggingface", "ai21labs/AI21-Jamba-Mini-1.6", device=backend.device)
    print("loaded model")

    print(f"type num_few_shots: {args.shots}")
    print(f"k: {args.k}")

    # have to enable prefill sparsification for benchmarks
    for k in args.k:
        model.enable_head_sparsification(k=k, metric="entropy", prefill=True)

        save_dir = Path("lm_eval_results")
        save_path = save_dir / f"{args.task}_k{k}.pkl"
        print(f"save path: {save_path}")

        model_wrapped = HFLM(model.model, tokenizer=model.tokenizer, batch_size="auto")
        print("running eval")
        results = simple_evaluate(
            model=model_wrapped,
            tasks=[args.task],
            num_fewshot=args.shots,
            batch_size="auto",
            apply_chat_template=True,
            fewshot_as_multiturn=True,
        )
        print("eval done")
        print(results)

        save_dir = Path("lm_eval_results")
        save_path = save_dir / f"{args.task}_k{k}.pkl"
        with open(str(save_path), "wb") as f:
            pickle.dump(results, f)
            print(f"saved results to {save_path}")
