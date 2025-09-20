# %%

from contextlib import AbstractContextManager

# %%


class SimpleContext(AbstractContextManager):
    def __enter__(self):
        print("Entering the context")
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        print("Exiting the context")


with SimpleContext() as ctx:
    print("Inside the context")


# %%


class Trace(AbstractContextManager):
    def __init__(self, func):
        self.func = func
        self.input = None
        self.output = None

    def __enter__(self):
        # Save the original function
        self._original_func = self.func

        def wrapper(*args, **kwargs):
            self.input = (args, kwargs)
            self.output = self._original_func(*args, **kwargs)  # Call original!
            return self.output

        self.func = wrapper
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.func = self._original_func


def add(a, b):
    return a + b


with Trace(add) as tr:
    result = tr.func(2, 3)
    print("Input: ", tr.input)
    print("Output: ", tr.output)
    print("Result: ", result)

# %%


class Model:
    def layer1(self, x):
        return x + 1

    def layer2(self, x):
        return x + 2

    def __call__(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return x


class LayerTrace(AbstractContextManager):
    def __init__(self, obj, attr_names, edit_output):
        self.obj = obj
        self.layer_names = attr_names
        self.outputs = {}
        self.edit_output = edit_output

    def __enter__(self):
        self._original_methods = {}
        for name in self.layer_names:
            orig = getattr(self.obj, name)
            self._original_methods[name] = orig

            def make_wrapper(layer_name, orig_func):
                def wrapper(*args, **kwargs):
                    result = orig_func(*args, **kwargs)
                    if self.edit_output:
                        result = self.edit_output(result)
                    self.outputs[layer_name] = result
                    return result

                return wrapper

            setattr(self.obj, name, make_wrapper(name, orig))
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        for name, orig in self._original_methods.items():
            setattr(self.obj, name, orig)


def edit_output_fn(x):
    return x * 10


model = Model()

with LayerTrace(model, attr_names=["layer1"], edit_output=edit_output_fn) as tr:
    result = model(2)
    print("Output: ", tr.outputs)
    print("Result: ", result)

# %%
