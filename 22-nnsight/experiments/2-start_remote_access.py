# %%

from dotenv import load_dotenv
load_dotenv("22-nnsight/.env")
import os

from nnsight import CONFIG
CONFIG.set_default_api_key(os.getenv("NDIF_API_KEY"))

from nnsight import LanguageModel
from torch import Tensor

# %% Choose a Model
#
# NDIF hosts multiple LLMs, including various sizes of the Llama 3.1 models and
# DeepSeek-R1 models. You can view the full list of hosted models on
# [our status page](https://nnsight.net/status/). All of our models are open
# for public use, except you need to apply for access to the Llama-3.1-405B models
#
# Details
# Apply for 405B access
# If you have a clear research need for Llama-3.1-405B and would like more details
# about applying for access, please refer to [this page](https://ndif.us/405b.html)!
#
# For these exercises, we will explore how we can access and modify the Llama-3.1-70B
# model's internal states. This 70-billion-parameter model is about the maximum
# size that you could run on a single A100 GPU with 80GB of VRAM, but we are
# going to access it remotely on NDIF resources, so you can run it on Colab or
# your laptop computer!
#
# Details
# Note: Llama models are gated on HuggingFace
#
# Llama models are gated and require you to register for access via HuggingFace.
# [Check out their website for more information about registration with Meta](https://huggingface.co/meta-llama/Llama-3.1-70B)
#
# If you are using a local Python installation, you can activate your HuggingFace token
# using the terminal:
#
# `huggingface-cli login -token YOUR_HF_TOKEN`
# 
# If you are using Colab, you can add your HuggingFace token to your Secrets.
# 
# We will be using the `LanguageModel` subclass of NNsight to load in the
# Llama-3.1-70B model and access its internal states.
# 
# Details
# About NNsight LanguageModel
#
# The `LanguageModel` subclass of NNsight is a wrapper that includes special support
# for HuggingFace language models, including automatically loading models from a
# HuggingFace ID together with the appropriate tokenizer.
#
# This way there's no need to pretokenize your input, and instead you can
# just pass a string as an input!
#
# Note: `LanguageModel` models also accept tokenized inputs, including
# [chat templates](https://huggingface.co/docs/transformers/main/en/chat_templating)

# instantiate the model using the LanguageModel class

# don't worry, this won't load locally!
llm = LanguageModel("deepseek-ai/DeepSeek-R1-Distill-Llama-8B", device_map="auto")

print(llm)

# %% Access model internals
#
# Now that we've installed `nnsight`, configured our API key, and instantiated a model,
# we can run an experiment.
#
# For this experiment, let's try grabbing some of the LLM's hidden states using
# `nnsight`'s tracing context, `.trace()`
# 
# Entering the tracing context allows su to customize how a neural network runs.
# By calling `.trace()`, we are telling the model to run with a given input
# and to collect and/or modify the internal model states based on user-defined
# code within the tracing context. We can also specify that we want to use
# NDIF-hosted model instead of executing locally by setting `remote=True`.
#
# To get started, let's ask NNsight to collect the layer output (known as "logits")
# at the final layer, along with the overall model output. NNsight needs to know
# what specific parts of the model we're interested in accessing later, so we need
# to specify which elements we'd like to save after exiting the tracing context
# using `.save()`.
#
# Note: You will not be able to access any values defined within a `.trace()`
# that aren't saved with `.save()` after exiting the tracing context!

# remote = True means the model will execute on NDIF's shared resources
with llm.trace("The Eiffel Tower is in the city of", remote=True):
    # user-defined code to access internal model components
    hidden_states: Tensor = llm.model.layers[-1].output[0].save()
    output: Tensor = llm.output.save()

# %%

# after exiting the tracing context, we can access any values that were saved
print("Hidden State Logits: ", hidden_states[0])

output_logits = output["logits"]
print("Model Output Logits: ", output_logits[0])

# decode the final model output from output logits
max_probs, tokens = output_logits[0].max(dim=-1)
word = [llm.tokenizer.decode(tokens.cpu()[-1])]
print("Model Output: ", word[0])

# What are we seeing here? NNsight tells you if your job 
# is received, approved, running, or completed via logs.
#
# Details
# Disabling remote logging notifications if you prefer,
# you can disable NNsight remote logging notifications
# with the following code, although they can help
# troubleshoot any network issues.
# from nnsight import CONFIG
# CONFIG.APP.REMOTE_LOGGING = False
#
# We are also seeing our printed results. After exiting
# the tracing context, NNsight downloads the saved results,
# which we can perform operations on using Python code.
# Pretty simple!

# %% Alter model internals
#
# Now that we've accessed the internal layers of the model,
# let's try modifying them and see how it affects the output!
#
# We can do this using in-place operations in NNsight,
# which alter the model's state during execution.
# Let's try changing the output of layer 8 to be equal to 4.

with llm.trace("The Eiffel Tower is in the city of", remote=True):
    
    # user-defined code to access internal model components
    llm.model.layers[7].output[0][:] = 4 # in-place operation to change a single layer's output values
    output = llm.output.save()

# after exiting the tracing context, we can access any
# values that were saved

output_logits = output["logits"]
print("Model Output Logits: ", output_logits[0])

# decode the final model output from output logits
max_probs, tokens = output_logits[0].max(dim=-1)
word = [llm.tokenizer.decode(tokens.cpu()[-1])]
print("Model Output: ", word[0])

# Okay! The output for "The Eiffel Tower is in the city of"
# is now "Destruction". Looks like our intervention on the hidden
# 8th layer worked to change the model output!

# %%
# 
# Are you ready for something a little more complicated?
# Let's take the model's state when answering the city
# that the London Bridge is in, and swap that into the model's final layer when
# answering the Eiffel Tower question! We can do this using NNsight's invoking contexts,
# which batch different inputs into the same run through the model.
#
# We can access values defiend in invoking contexts throughout the other invoke
# context, allowing us to do something like swapping model tates for different
# inputs. Let's try it out!

with llm.trace(remote=True) as tracer:
    with tracer.invoke("The London Bridge is in the city of"):
        hidden_states = llm.model.layers[-1].output[0] # no .save()
    
    with tracer.invoke("The Eiffel Tower is in the city of"):
        # user-defined code to access internal model components
        llm.model.layers[-1].output[0][:] = hidden_states # can be accessed without .save()!
        output = llm.output.save()

output_logits = output["logits"]
print("Model Output Logits: ", output_logits[0])

# decode the final model output from output logits
max_probs, tokens = output_logits[0].max(dim=-1)
word = [llm.tokenizer.decode(tokens.cpu()[-1])]
print("Model Output: ", word[0])

# Awesome, looks like it worked! The model output London instead of Paris when
# asked about the location of the Eiffel Tower.

# %% Next steps: Run your own experiment with NDIF and NNsight
#
# This is just a quick overview of some of NNsight's functionality when working
# with remote models, so to learn more we recommend taking a deeper dive into
# these resources:
# - Get a comprehensive overview of the library with the 
# [NNsight Walkthrough](https://nnsight.net/notebooks/tutorials/walkthrough/)
# - Check out some NNsight implementations of common 
# [LLM interpretability techniques](https://nnsight.net/tutorials/)
# Join the conversation with the NDIF [Discord](https://discord.com/invite/6uFJmCSwW7) 
# community
# - Follow us on [GitHub](https://github.com/ndif-team/nnsight), 
# [Bluesky](https://bsky.app/profile/ndif-team.bsky.social), 
# [X](https://x.com/ndif_team), and [LinkedIn](https://www.linkedin.com/company/national-deep-inference-fabric/)
#
# Want to scale up your research? 
# [Apply for access to Llama-3.1-405B](https://ndif.us/405b.html)