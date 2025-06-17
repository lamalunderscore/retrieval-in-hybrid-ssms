import os
import subprocess
from pathlib import Path


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

    # Run NIAH workflow
    run_niah(prompt=True, pred=False, eval=False, vis=False)


if __name__ == "__main__":
    main()
