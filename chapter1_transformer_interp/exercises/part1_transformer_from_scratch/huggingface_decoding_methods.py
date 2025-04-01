# %%


from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

torch_device = "cuda" if torch.cuda.is_available() else "cpu"

MAIN = __name__ == "__main__"

# %%


if MAIN:
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    # add the EOS token as PAD token to avoid warnings
    model = AutoModelForCausalLM.from_pretrained("gpt2", pad_token_id=tokenizer.eos_token_id).to(torch_device)

# %%


if MAIN:
    # encode context the generation is conditioned on
    model_inputs = tokenizer("I enjoy walking with my cute dog", return_tensors="pt").to(torch_device)

    # generate 40 new tokens
    greedy_output = model.generate(**model_inputs, max_new_tokens=40)

    print("Output:\n" + 100 * '-')
    print(tokenizer.decode(greedy_output[0], skip_special_tokens=True))

# %%


if MAIN:
    beam_output = model.generate(
        **model_inputs,
        max_new_tokens=40,
        num_beams=5,
        early_stopping=True)

    print("Output:\n" + 100 * '-')
    print(tokenizer.decode(beam_output[0], skip_special_tokens=True))

# %%


if MAIN:
    beam_output = model.generate(
        **model_inputs,
        max_new_tokens=40,
        num_beams=5,
        no_repeat_ngram_size=2,
        early_stopping=True
    )

    print("Output:\n" + 100 * '-')
    print(tokenizer.decode(beam_output[0], skip_special_tokens=True))

# %%


if MAIN:
    # set return_num_sequences > 1
    beam_outputs = model.generate(
        **model_inputs,
        max_new_tokens=40,
        num_beams=5,
        no_repeat_ngram_size=2,
        num_return_sequences=5,
        early_stopping=True
    )

    # now we have 3 output sequences
    print("Output:\n" + 100 * '-')
    for i, beam_output in enumerate(beam_outputs):
        print("{}: {}".format(i, tokenizer.decode(beam_output, skip_special_tokens=True)))

# %%


if MAIN:
    # set seed to reproduce results. Feel free to change the seed though to get different results
    from transformers import set_seed
    set_seed(42)

    # activate sampling and deactivate top_k by setting top_k sampling to 0
    sample_output = model.generate(
        **model_inputs,
        max_new_tokens=40,
        do_sample=True,
        top_k=0
    )

    print("Output:\n" + 100 * '-')
    print(tokenizer.decode(sample_output[0], skip_special_tokens=True))

# %%


if MAIN:
    # set seed to reproduce results. Feel free to change the seed though to get different results
    set_seed(42)

    # use temperature to decrease the sensitivity to low probability candidates
    sample_output = model.generate(
        **model_inputs,
        max_new_tokens=40,
        do_sample=True,
        top_k=0,
        temperature=0.6,
    )

    print("Output:\n" + 100 * '-')
    print(tokenizer.decode(sample_output[0], skip_special_tokens=True))

# %%


if MAIN:
    # set seed to reproduce results. Feel free to change the seed though to get different results
    set_seed(42)

    # set top_k to 50
    sample_output = model.generate(
        **model_inputs,
        max_new_tokens=40,
        do_sample=True,
        top_k=50
    )

    print("Output:\n" + 100 * '-')
    print(tokenizer.decode(sample_output[0], skip_special_tokens=True))


# %%


if MAIN:
    # set seed to reproduce results. Feel free to change the seed though to get different results
    set_seed(42)

    # set top_k to 50
    sample_output = model.generate(
        **model_inputs,
        max_new_tokens=40,
        do_sample=True,
        top_p=0.92,
        top_k=0
    )

    print("Output:\n" + 100 * '-')
    print(tokenizer.decode(sample_output[0], skip_special_tokens=True))

# %%


if MAIN:
    # set seed to reproduce results. Feel free to change the seed though to get different results
    set_seed(42)

    # set top_k = 50 and set top_p = 0.95 and num_return_sequences = 3
    sample_outputs = model.generate(
        **model_inputs,
        max_new_tokens=40,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        num_return_sequences=3,
    )

    print("Output:\n" + 100 * '-')
    for i, sample_output in enumerate(sample_outputs):
        print("{}: {}".format(i, tokenizer.decode(sample_output, skip_special_tokens=True)))

# %%


