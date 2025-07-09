# Timer

1. Evaluations

    1. Set up environment start time: 14.41
    2. Code start time: 15.04
    3. Set up environment start time (reason: OpenAI is not reachable): 15.38
    4. Code start time: 16.15
    5. Code end time: 16.38
    6. Save output end time (reason: can't commit with openai api key. the time was spent to refactor): 16.59

# Installation

1. Create conda env

    ```bash
    conda create --name algoverse python=3.11
    conda activate algoverse
    ```

2. Install pythonn packages

    ```bash
    pip install -r requirements.txt
    ```

## Evaluations

3. Download the ARC-Challenge dataset: https://huggingface.co/datasets/allenai/ai2_arc

    Change the `WORKSPACE_PATH` variable value

    ```bash
    WORKSPACE_PATH=/Users/jason/Documents
    HF_ENDPOINT=https://hf-mirror.com huggingface-cli download allenai/ai2_arc --repo-type dataset --local-dir "$WORKSPACE_PATH/datasets/allenai/ai2_arc"
    ```

4. Run the `main.py`

    Change the `WORKSPACE_PATH` and `OPENAI_API_KEY` variable

    ```bash
    WORKSPACE_PATH=/Users/jason/Documents
    OPENAI_API_KEY=YOUR_API_KEY
    python main.py --dataset_path "$WORKSPACE_PATH/datasets" --openai_api_key "$OPENAI_API_KEY" > output.txt
    ```