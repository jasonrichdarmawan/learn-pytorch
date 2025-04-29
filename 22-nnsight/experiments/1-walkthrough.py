# %%

from collections import OrderedDict
import torch

import nnsight
from nnsight import NNsight, LanguageModel

# %% NNsight - Tracing Context
# Reference: https://nnsight.net/notebooks/tutorials/walkthrough/

input_size = 5
hidden_dims = 10
output_size = 2

net = torch.nn.Sequential(
    OrderedDict(
        [
            ("layer1", torch.nn.Linear(input_size, hidden_dims)),
            ("layer2", torch.nn.Linear(hidden_dims, output_size)),
        ]
    )
).requires_grad_(False)

# %%

tiny_model = NNsight(net)
print(tiny_model)

# %%

input = torch.rand((1, input_size))

with tiny_model.trace(input) as tracer:
    output = tiny_model.output.save()
    # it is important to understand that the model is not executed until the end of tracing 
    # context.
    # How can we access inputs and outputs before the model is run? The trick is deferred 
    # execution.
    #
    # `.input` and `.output` are Proxies for the ventual inputs and outputs of a module.
    # In other words, when we access `model.output` what we are communicating to nnsight is,
    # "When you compute the output of `model`, please grab it for me and put the value into
    # its corresponding Proxy object"
    #
    # Proxy objects will only have their value at the end of a context if we call .save()
    # on them. This helps to reduce memory costs. Adding .save() fixes the error

print(output)

# %%

with tiny_model.trace(input) as tracer:
    l1_output = tiny_model.layer1.output.save()

print(l1_output)

# %%

output = tiny_model.trace(input, trace=False)

print(output)

# %%

with tiny_model.trace(input):

    l2_input = tiny_model.layer2.input.save()
    # On module inputs
    #
    # Notice how the value for l2_input is just a single tensor. By default, the `.input`
    # attribute of a module will return the first tensor input to the module.
    #
    # We can also access the full input to a module by using the `.inputs` attribute,
    # which will return the values in the form of:
    # ```
    # tuple(tuple(args), dictionary(kwargs))
    # ```
    # Where the first index of the tuple is itself a tuple of all positional arguments,
    # and the second index is a dictionary of all keyword arguments.

print(l2_input)

# %%
# Until now we were saving the output of the model and its submodules within the
# `Trace` context to then print it after exiting the context. We will continuing
# doing this in the rest of the tutorial since it's a good practice to save the
# computation results for later analysis.
#
# However, we can also log the outputs of the model and its submodules within the 
# `Trace` context. This is useful for debugging and understanding the model's
# behavior while saving memory.

with tiny_model.trace(input) as tracer:
    tracer.log("Layer 1 - out: ", tiny_model.layer1.output)

# %%
# Now that we can access activations, we also want to do some post-processing on it.
# Let's find out which dimension of layer1's output has the highest value.
#
# We could do this by calling torch.argmax(...) after the tracing context
# or we can just leverage the fact that nnsight handles Pytorch functions and methods
# within the tracing context, by creating a Proxy request for it:

with tiny_model.trace(input):
    
    # Note we don't need to call .save() on the output,
    # as we're only using its value within the tracing context.
    l1_output = tiny_model.layer1.output

    # We do need to save the argmax tensor however,
    # as we're using it outside the tracing context.
    l1_amax = torch.argmax(l1_output, dim=1).save()

print(l1_amax[0])

# %%

with tiny_model.trace(input):
    value = (tiny_model.layer1.output.sum() + tiny_model.layer2.output.sum()).save()

print(value)
# The code block above is saying to `nnsight`, "Run the model with the given input. 
# When the output of `tiny_model.layer1` is computed, take its sum. Then do the same
# for `tiny_model.layer2`. Now that both of those are computed, add them and make
# sure not to delete this value as I wish to use it outside of the tracing context."

# %%
# Custom Functions
# Everything within the tracing cotnext operates on the intervention graph. Therefore,
# for `nnsight` to trace a function it must also be a part of the intervention graph.
#
# Out-of-the-box `nnsight` supports PyTorch functions and methods, all operators, as well
# the `einops` library. We don't need to do anything special to use them.
# But what do we do if we want to use custom functions? How do we add them to the
# intervention graph?
#
# Enter `nnsight.apply()`. It allows us to add new functions to the intervention graph.
# Let's see how it works:

# Take a tensor and return the sum of its elements
def tensor_sum(tensor: torch.Tensor):
    flat = tensor.flatten()
    total = 0
    for element in flat:
        total += element.item()

    return torch.tensor(total)

with tiny_model.trace(input) as tracer:

    # Specify the function name and its arguments (in a comma-separated form) 
    # to add to the intervention graph
    custom_sum = nnsight.apply(tensor_sum, tiny_model.layer1.output).save()
    sum = tiny_model.layer1.output.sum()
    sum.save()

print(custom_sum, sum)
# `nnsight.apply()` executes the function it wraps and returns its output as a Proxy object.
# We can then use this Proxy object as we would any other.
#
# The aplications of `nnsight.apply` are wide: it can be used to wrap any custom function
# or functions from libraries that `nnsight` does not support out-of-the-box.

# %%
# Setting
#
# Getting and analyzing the activations from various points in a model can be really
# insightful, and a number of ML techniques do exactly that. However, of ten we not only
# want to view the computation of a model, but also to influence it.
# 
# To demonstrate the effect of editing the flow of information through the model,
# let's set the first dimension of the first layer's output to 0. `NNsight` makes this
# really easy using the `=` operator:

with tiny_model.trace(input):
    
    # Save the output before the edit to compare.
    # Notice we apply .clone() before saving as the setting operation is in-place.
    l1_output_before = tiny_model.layer1.output.clone().save()

    # Access the 0th index of the hidden state dimension and set it to 0.
    tiny_model.layer1.output[:, 0] = 0

    # Save the output after to see our edit.
    l1_output_after = tiny_model.layer1.output.save()

print("Before:", l1_output_before)
print("After:", l1_output_after)

# %%

with tiny_model.trace(input):
    
    # Save the output before the edit to compare.
    # Notice we apply .clone() before saving as the setting operation is in-place.
    l1_output_before = tiny_model.layer1.output.clone().save()

    # Access the last index of the hidden state dimension and set it to 0.
    tiny_model.layer1.output[:, hidden_dims-1] = 0

    # Save the output after to see our edit.
    l1_output_after = tiny_model.layer1.output.save()

print("Before:", l1_output_before)
print("After:", l1_output_after)

# Oh no, we are gettign an error! Ah of course, we needed to index at `hidden_dims - 1`
# not `hidden_dims`.
#
# if you've been using `nnsight`, you are probably familiar with error messages that can
# be quite difficult to troubleshoot. In `nnsight 0.4` we've now improved error messaging
# to be descriptive and line-specific, as you should see in the above example:

# %% Scanning and Validating
# Error codes are helpful, but sometimes you may want to quickly troubleshoot
# your code without actually running it.
#
# Enter "Scanning" and "Validating"! We can enable this features by setting the `scan=True`
# and `validate=True` flag in the `trace` method.
#
# "Scanning" runs "fake" inputs through the model to collect information like shapes and types
# (i.e., scanning will populate all called .inputs and .outputs)
#
# "Validating" attempts to execute the intervention proxies with "fake" inputs to check
# if they work (i.e., executes all intervention in your code with fake tensors).
#
# "Validating" is dependent on "Scanning" to work correctly, so we need to run the
# scan of the model at least once to debug with validate. Let's try it out on our example code

with tiny_model.trace(input, scan=True, validate=True):
    l1_output_before = tiny_model.layer1.output.clone().save()

    # the error is happening here
    tiny_model.layer1.output[:, hidden_dims-1] = 0

    l1_output_after = tiny_model.layer1.output.save()

print("Before:", l1_output_before)
print("After:", l1_output_after)

# The operations are never executed using tensors with real values so it doesn't incur any
# memory costs. Then, when creating proxy requests like the setting one above, `nnsight`
# also attempts to execute the request on the "fake" values we recorded. Hence, it lets
# us know if our request is feasible before even running the model.
# [Here](https://nnsight.net/notebooks/features/scan_validate/) is a more detailed example 
# of scan and validate in action
# 
# Details
# A word of caution
# Some pytorch operations and related libraries don't work well with fake tensors
# If you are doing anything in a loop where efficiency is important, you should
# keep scanning and validating off. It's best to use them only when debugging or
# when you are unsure if your intervention will work

# %%
# We can also use the `.scan()` method to get the shape of a module without having
# to fully run the model. If scan is enabled, our input is run through the model
# under its own "fake" context. This means the input makes its way through all of the model
# operations, allowing `nnsight` to record the shapes and data types of module inputs and outputs!

with tiny_model.scan(input):
    dim = tiny_model.layer1.output.shape[-1]

print(dim)

# %%
# `LanguageModel` is a subclass of `NNsight`. While we could define and create a model
# # to pass in directly, `LanguageModel` includes special support for HuggingFace 
# language models, including automatically loading models from a Huggingface ID,
# and loading the model together with the appropriate tokenizer.
# Here is how we can use `LanguageModel` to load `GPT-2`

llm = LanguageModel("openai-community/gpt2", device_map="auto")

print(llm)
# When we initialize `LanguageModel`, we aren't yet loading the parameters of the model
# into memory. We actually loading a `meta` version of the model which doesn't take
# up any memory, but still allows us to view and trace actions on it.
# After exiting the first tracing context, the model is then fully loaded into memory.
# To load into memory on initialization, you can pass
# `dispatch=True` into `LanguageModel` like 
# `LanguageModel('openai-community/gpt2', device_map="auto", dispatch=True)`.
#
# Details
# On Model Initialization
# 
# A few important things to note:
# 
# Keyword arguments passed to the initialization of `LanguageModel` is forwaded to
# HuggingFace specific loading logic. In this case, `device_map` specifies which devices
# to use and its value `auto` indicates to evenly distribute it to all available GPUs
# (and CPU if no GPUs available). Other arguemnts can be found here:
# https://huggingface.co/docs/transformers/model_doc/auto#transformers.AutoModelForCausalLM

# %%
# Let's now apply some of the features that we used on the small model to `GPT-2`.
# Unlike `NNsight`, `LanguageModel` does not define logic to pre-process inputs
# upon entering the tracing context. This makes interacting with the model simpler
# (i.e., you can send prompts to the model without having to directly access the
# tokenizer).
#
# In the following example, we ablate the value coming from the last layer's MLP
# module and decode the logits to see what token them odel predicts without influence from
# that particular module:

with llm.trace("The Eiffel Tower is in the city of"):

    # Access the last layer using h[-1] as it's a ModuleList
    # Access the first index of .output as that's where the hidden states are.
    llm.transformer.h[-1].mlp.output[0][:] = 0

    # Logits come out of model.lm_head and we apply argmax to get the predicted token ids.
    token_ids = llm.lm_head.output.argmax(dim=-1).save()

print("\nToken IDs:", token_ids)

# Apply the tokenizer to decode the ids into words after the tracing context.
print("Prediction:", llm.tokenizer.decode(token_ids[0][-1]))

# We just ran a little intervention on a much mroe complex model with many more parameters!
# However, we're missing an important piece of information: what the prediction would have
# looked witohut our ablation.
#
# We could just run two tracing contexts and compare the outputs. However, this would require
# two forward passes through the model. `NNsight` can do better than that with batching.

# %% Batching
# Batching is a  way to process multiple inputs in one forward pass. To better understand
# how batching works, we're going to bring back the `Tracer` object that we dropped before.
# 
# When we call `.trace(...)`, it's actually creating two different contexts behind the 
# scenes. The first one is the tracing context that we've discussed previously, 
# and the second one is the invoker context. 
# The invoker context defines the values of the `.input` and `.output` Proxies.
# 
# If we call .`trace(...)` with some input, the input is passed on to the invoker.
# As there is only one input, only one invoker context is created.
# 
# If we call `.trace(...)` without an input, then we can call `tracer.invoke(input1)`
# to manually create the invoker context with an input, `input1`. We can also repeatedly
# call `tracer.invoke(...)` to create the invoker context for additional inputs.
# Every subsequent time we call `.invoke(...)`, interventions within its context
# will only refer to the input in that particular invoke statement.
#
# When exiting the tracing context, the inputs from all of the invokers will be batched
# together, and they will be executed in one forward pass! To test this out, let's do
# the same ablation experiment, but also add a `control` output for comparison:
#
# Details
# More on the invoker context
#
# Note that when injecting data to only the relevant invoker interventions, `nnsight` tries,
# but can't guarantee, to narrow the data into the right batch indices. Thus, there are 
# cases where all invokes will get all of the data. Specifically, if the input or output
# data is stored as an object that is not an arbitrary collection of tensors, it will be
# broadcsted to all invokes.
#
# Just like `.trace(...)` created a `Tracer` object, `.invoke(...)` creates an `Invoker` object.
# For `LanguageModel`, the `Invoker` prepares the input by running a tokenizer on it.
# `Invoker` stores pre-processed inputs at `invoker.inputs`, which can be accessed
# to see information about our inputs. In a case where we pass a single input to 
# `.trace(...)` directly, we can still access the invoker object at `tracker.invoker`
# without having to call `tracer.invoke(...)`.
#
# Keyword arguments given to `.invoke(...)` make their way to the input pre-processing.
# `LanguageModel` has keyword arguments `max_length` and `truncation` used for tokenization
# which can be passed to the invoker. If we want to pass keyword arguments to the invoker
# for a single input `.trace(...)`, we can pass `invoker_args` as a dictionary of invoker
# keyword arguments.
# Here is an example to demonstrate everything we've described:

# This snippet
with llm.trace("hello", invoker_args={"max_length": 10}) as tracer:
    invoker = tracer.invoker

# does the same as
with llm.trace() as tracer:
    with tracer.invoke("hello", max_length=10) as invoker:
        invoker = invoker

with llm.trace() as tracer:
    with tracer.invoke("The Eiffel Tower is in the city of"):

        # Ablate the last MLP for only this batch.
        llm.transformer.h[-1].mlp.output[0][:] = 0

        # Get the output for only the intervened on batch.
        token_ids_intervention = llm.lm_head.output.argmax(dim=-1).save()

    with tracer.invoke("The Eiffel Tower is in the city of"):

        # Get the output for only the original batch.
        token_ids_original = llm.lm_head.output.argmax(dim=-1).save()
    
print("Original token IDs:", token_ids_original)
print("Modified token IDs:", token_ids_intervention)

print("Original prediction:", llm.tokenizer.decode(token_ids_original[0][-1]))
print("Modified prediction:", llm.tokenizer.decode(token_ids_intervention[0][-1]))

# Based on our control results, our ablation did end up affecting what the model predicted.
# That's pretty neat

# %%
# Another cool thing with multiple invokes is that Proxies can interact between them.
#
# Here, we transfer the token embeddings from a real prompt into another placeholder prompt.
# Therefore the latter prompt produces the output of the former prompt.
with llm.trace() as tracer:
    with tracer.invoke("The Eiffel Tower is in the city of"):
        embeddings = llm.transformer.wte.output
    
    with tracer.invoke("_ _ _ _ _ _ _ _ _ _"):
        llm.transformer.wte.output = embeddings
        token_ids_intervention = llm.lm_head.output.argmax(dim=-1).save()
    
    with tracer.invoke("_ _ _ _ _ _ _ _ _ _"):
        token_ids_original = llm.lm_head.output.argmax(dim=-1).save()

print("original prediction shape", token_ids_original[0][-1].shape)
print("Original prediction:", llm.tokenizer.decode(token_ids_original[0][-1]))

print("modified prediction shape", token_ids_intervention[0][-1].shape)
print("Modified prediction:", llm.tokenizer.decode(token_ids_intervention[0][-1]))

# For larger batch sizes, you can also iterate across multiple invoke cotnexts.

# %% Multiple Token Generation
# .next()
#
# Some HuggignFace models define methods to generate multiple outputs
# at a time. `LanguageModel` wraps that functionality to provide
# the same tracing features by using `.generate(...)` instead of
# `.trace(...)`. This calls the underlying models'
# `.generate` method. It passes the output through a `.generator`
# module that we've added onto the model, allowing us to get the
# generate output at `.generator.output`.
#
# In a case like this, the underlying model is called more than once;
# the modules of said model produce more than one output.
# Which iteration should a given `module.output` refer to?
# That's where `Module.next()` comes in!
# 
# Each module has a call index associated with it and `.next()` simply
# increments that attribute. At the time of execution, data is injected
# into the itnervention graph only at the iteration that matches the
# call index.

with llm.generate("The Eiffel Tower is in the city of", max_new_tokens=3) as tracer:
    
    hidden_states1 = llm.transformer.h[-1].output[0].save()

    # use module.next() to access the next intervention
    hidden_states2 = llm.transformer.h[-1].next().output[0].save()

    # saving the output allows you to save the hidden state across the initial prompt
    out = llm.generator.output.save()

print("hidden_states1.shape", hidden_states1.shape)
print("hidden_states2.shape", hidden_states2.shape)
print("out.shape", out.shape)
print(f"out:\n{out}")

# %% Gradients
# `NNsight` also lets us apply backpropagation and access gradients with respect to loss.
# Like `.input` and `.output` on modules, `nnsight` exposes `.grad` on Proxies themselves
# (assuming they are proxies of tensors):

input = torch.rand((2, input_size))

with tiny_model.trace(input):
    
    # We need to explicitly have the tensor require grad
    # as the model we defined earlier turned off requiring grad.
    tiny_model.layer1.output.requires_grad = True

    # We call .grad on tensor Proxy to communicate we want to store its gradient.
    # We need to call .save() since .grad is its own Proxy.
    layer1_output_grad = tiny_model.layer1.output.grad.save()
    layer2_output_grad = tiny_model.layer2.output.grad.save()

    # Need a loss to propagate through the later modules in order to have a grad.
    loss = tiny_model.output.sum()
    loss.backward()

print("Layer 1 output gradient:", layer1_output_grad)
print("Layer 2 output gradient:", layer2_output_grad)

# %%
