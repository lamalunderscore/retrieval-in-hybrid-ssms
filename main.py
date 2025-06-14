import os
import subprocess
from pathlib import Path

import kagglehub
import yaml


# directory for NIAH scripts
NIAH_DIR = Path("NIAH/Needle_test")

# the config file inside the NIAH directory
CONF_FILE = "config.yaml"


def run_command(command):
    """Run a command in a subprocess and prints its output (stdout and stderr) as it is generated.

    Args:
        command (list): The command to execute, provided as a list of strings (e.g., ["ping", "google.com"]).

    """
    env = os.environ.copy()
    print(f"CUDA_VISIBLE_DEVICES before adjustment: {env['CUDA_VISIBLE_DEVICES']}")
    env["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4"

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


def run_niah(prompt=True, pred=True, eval=True, vis=True):
    """Run the NIAH (Needle In A Haystack) workflow.

    Args:
        prompt (bool): If the prompt script should be run.
        pred (bool): If the pred script should be run.
        eval (bool): If the eval script should be run.
        vis (bool): If the vis script should be run.

    """
    print("Running NIAH workflow...")

    # Use your own HF_token here or set it as an environment variable
    if not os.environ.get("HF_TOKEN") and not os.environ.get("HUGGINGFACE_TOKEN"):
        print("No token found in environment variables, using predefined token...")
        os.environ["HF_TOKEN"] = "your_huggingface_token"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    # Run NIAH scripts
    if prompt:
        print("Running prompt.py...")
        run_command(["python", "-u", NIAH_DIR / "prompt.py"])

    if pred:
        print("Running predictions...")
        run_command(["python", "-u", NIAH_DIR / "pred.py"])

    if eval:
        print("Running evaluation...")
        run_command(["python", "-u", NIAH_DIR / "eval.py"])

    if vis:
        print("Running visualisation...")
        run_command(["python", "-u", NIAH_DIR / "vis.py"])

    print("NIAH workflow completed")


def main():
    # MAKE SURE TO AUTHENTICATE KAGGLE BEFORE
    # https://github.com/Kaggle/kagglehub?tab=readme-ov-file#authenticate

    if not os.environ.get("KAGGLE_KEY") or not os.environ.get("KAGGLE_USERNAME"):
        print("No token found in environment variables, using predefined token...")
        os.environ["HF_TOKEN"] = "your_huggingface_token"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    conf_path = NIAH_DIR / CONF_FILE
    print(f"conf_path: {conf_path}")
    with open(conf_path, "r") as f:
        config = yaml.safe_load(f)

    if config["pred"]["model_type"] == "recurrentgemma":
        # Download model
        print("Downloading model...")
        variant = "2b"  # 2b/9b available
        model_dir = Path(kagglehub.model_download(f"google/recurrentgemma/PyTorch/{variant}"))
        model_path = model_dir / f"{variant}.pt"
        tokenizer_path = model_dir / "tokenizer.model"
        print(model_path)

        print("Updating config...")
        # set path in config
        config["pred"]["model_path"] = str(model_path)
        config["pred"]["tokenizer_path"] = str(tokenizer_path)
        dump = yaml.dump(config)

        with open(conf_path, "w") as f:
            f.write(dump)

    # Run NIAH workflow
    run_niah(prompt=False, pred=True, eval=False, vis=False)


if __name__ == "__main__":
    main()
