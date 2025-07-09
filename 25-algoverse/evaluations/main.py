# %%

import sys
from argparse import ArgumentParser
import os
from datasets import load_dataset
from openai import OpenAI
from tqdm import tqdm
import re

# %%

print("Setting up environment...")

if False:
  print("Only use this in a Jupyter Notebook")
  print("Simulating environment setup...")
  WORKSPACE_PATH = "/Users/jason/Documents"
  sys.argv = [
    "main.py",
    "--dataset_path", f"{WORKSPACE_PATH}/datasets",
    "--openai_api_key", "YOUR_OPENAI_API_KEY",
    "--model_id", "ft:gpt-4.1-nano-2025-04-14:algoverse:arc-100-v1:BiRcF3ec",
    "--temperature", "0"
  ]

parser = ArgumentParser()
parser.add_argument(
  "--dataset_path",
  type=str,
  help="Path to the dataset directory."
)
parser.add_argument(
  "--openai_api_key",
  type=str,
  help="OpenAI API key."
)
parser.add_argument(
  "--model_id",
  type=str,
  help="Model ID to use for generating completions.",
  default="ft:gpt-4.1-nano-2025-04-14:algoverse:arc-100-v1:BiRcF3ec"
)
parser.add_argument(
  "--temperature",
  type=float,
  help="Temperature for the model's output.",
  default=0.0,
)
args = parser.parse_args().__dict__

# %%

print("Loading dataset...")
val_data = load_dataset(
  path=os.path.join(
    args["dataset_path"],
    "allenai/ai2_arc",
  ),
  name="ARC-Challenge",
  split="validation"
)
print("Data structure:")
print(val_data)

# %% 

print("Creating prompts...")
def create_prompt(item) -> str:
  """
  Supported datasets: allenai/ai2_arc/ARC-Challenge
  """
  prompt = f"Question {item['question']}\n"
  for label, choice in zip(
    item["choices"]["label"], item["choices"]["text"]
  ):
    prompt += f"{label}. {choice}\n"
  prompt += f"Answer: "

  return prompt

prompts = []
for item in val_data:
  prompts.append(create_prompt(item))

print("Prompt example:")
print(prompts[0])

# %%

print("Initializing OpenAI client...")
client = OpenAI(
  api_key=args["openai_api_key"],
)

# %%

def generate_chat_completion(
  client: OpenAI,
  prompt: str,
  model_id: str,
  temperature: float,
) -> str:
  chat_completion = client.chat.completions.create(
    messages=[
      {
        "role": "user",
        "content": prompt,
      },
    ],
    model=model_id,
    temperature=temperature,
  )
  return chat_completion

chat_completion = generate_chat_completion(
  client=client,
  prompt=prompts[0],
  model_id=args["model_id"],
  temperature=args["temperature"],
)
print("Chat completion example:")
print(chat_completion)

def extract_answer(response) -> str:
  content = response.choices[0].message.content
  return content

answer = extract_answer(chat_completion)
print("Extracted answer:")
print(answer)

# %%

print("Generating answers for all prompts...")
answers = []
for prompt in tqdm(prompts, desc="Generating answers"):
  response = generate_chat_completion(
    client=client,
    prompt=prompt,
    model_id=args["model_id"],
    temperature=args["temperature"],
  )
  answer = extract_answer(response)
  answers.append(answer)

# %%

count = len(val_data)
correct_count = 0
for item, answer in zip(val_data, answers):
  if answer.strip() == item["answerKey"]:
    correct_count += 1
accuracy = correct_count / count
print(f"Accuracy: {accuracy:.2%} ({correct_count}/{count})")

# %%

wrong_indices = []
for i, (item, answer) in enumerate(zip(val_data, answers)):
  if answer.strip() != item["answerKey"]:
    wrong_indices.append(i)

print("Answer indices that are wrong:")
print(wrong_indices)

# %%

print("Wrong answer example:")
print(f"Prompt:\n{prompts[wrong_indices[0]]}")
print(f"Expected answer: {val_data[wrong_indices[0]]['answerKey']}")
print(f"Generated answer: {answers[wrong_indices[0]]}")

# %%

def word_count(
  text: str,
  word_freq: dict[str, int] | None = None,
) -> dict[str, int]:
  words = re.findall(r'\b\w+\b', text.lower())
  if word_freq is None:
    word_freq = {}
  for word in words:
    word = word.lower()
    if word not in word_freq:
      word_freq[word] = 0
    word_freq[word] += 1
  return word_freq

print("Word count example:")
print(word_count(val_data[0]["question"]))

# %%

word_freq = {}
for wrong_index in wrong_indices:
  word_freq = word_count(
    text=val_data[wrong_index]["question"],
    word_freq=word_freq,
  )

print("Top 5 most frequent words and their counts in descending order:")
sorted_word_freq = sorted(
  word_freq.items(),
  key=lambda x: x[1],
  reverse=True
)
for word, count in sorted_word_freq[:5]:
  print(f"{word}: {count}")

# %%
