"""Script to coordinate NIAH experiments."""

import os
import subprocess
from argparse import ArgumentParser
from pathlib import Path

from NIAH.Needle_test.eval import run_eval
from NIAH.Needle_test.pred import run_predictions
from NIAH.Needle_test.prompt import run_prompts
from NIAH.Needle_test.vis import run_vis


# directory for NIAH scripts
NIAH_DIR = Path("NIAH/Needle_test")

# the config file inside the NIAH directory
CONF_FILE = "config.yaml"

parser = ArgumentParser(description="A script to coordinate the NIAH experiment.")
parser.add_argument("config", help="Name of the user", default=CONF_FILE)
parser.add_argument("--prompt", help="If prompts script should be run", action="store_true")
parser.add_argument("--pred", help="If pred script should be run", action="store_true")
parser.add_argument("--eval", help="If eval script should be run", action="store_true")
parser.add_argument("--vis", help="If vis script should be run", action="store_true")


def run_command(command):
    """Run a command in a subprocess and prints its output (stdout and stderr) as it is generated.

    Args:
        command (list): The command to execute, provided as a list of strings (e.g., ["ping", "google.com"]).

    """
    env = os.environ.copy()
    print(f"CUDA_VISIBLE_DEVICES before adjustment: {env['CUDA_VISIBLE_DEVICES']}")

    # Start the subprocess
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        env=env,
        text=True,
    )
    assert process.stdout is not None

    # Continuously read and print from stdout and stderr
    try:
        while True:
            # Read a line from stdout
            output = process.stdout.readline()
            if output:
                print(f"stdout: {output.strip()}")

            # Check if the process has finished and there's no more output
            if output == "" and process.poll() is not None:
                break

    finally:
        # Close the pipes and wait for the process to finish
        process.stdout.close()
        process.wait()


def run_niah(prompt=True, pred=True, eval=True, vis=True, config_file: str = CONF_FILE):
    """Run the NIAH (Needle In A Haystack) workflow.

    Args:
        prompt (bool): If the prompt script should be run.
        pred (bool): If the pred script should be run.
        eval (bool): If the eval script should be run.
        vis (bool): If the vis script should be run.
        config_file (str, *optional*): The config file to be used.

    """
    print("Running NIAH workflow...")

    # Use your own HF_token here or set it as an environment variable
    if not os.environ.get("HF_TOKEN") and not os.environ.get("HUGGINGFACE_TOKEN"):
        print("No token found in environment variables, using predefined token...")
        os.environ["HF_TOKEN"] = "your_huggingface_token"

    # Run NIAH scripts
    if prompt:
        print("Running prompt.py...")
        run_prompts(config_file)

    if pred:
        print("Running predictions...")
        run_predictions(config_file)

    if eval:
        print("Running evaluation...")
        run_eval(config_file)

    if vis:
        print("Running visualisation...")
        run_vis(config_file)

    print("NIAH workflow completed")


def main():  # noqa
    # MAKE SURE TO AUTHENTICATE KAGGLE BEFORE
    # https://github.com/Kaggle/kagglehub?tab=readme-ov-file#authenticate

    args = parser.parse_args()

    if not os.environ.get("KAGGLE_KEY") or not os.environ.get("KAGGLE_USERNAME"):
        print("No token found in environment variables, using predefined token...")
        os.environ["HF_TOKEN"] = "your_huggingface_token"

    # Run NIAH workflow
    run_niah(config_file=args.config, prompt=args.prompt, pred=args.pred, eval=args.eval, vis=args.vis)


if __name__ == "__main__":
    main()
