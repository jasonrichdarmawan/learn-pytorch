# %%

MAIN = "__main__" == __name__

from typing import Union

# %%

def process_list(items: list[Union[float, int]]) -> Union[float, None]:
    if not items: # Corner case: empty list
        print("List is empty, cannot process.")
        return None
    # ... proceed with processing ...
    return sum(items)

# Test cases
if MAIN:
    print(process_list([]))  # Should handle empty list
    print(process_list([1, 2, 3]))  # Should return 6
    print(process_list([0, 0, 0]))  # Should return 0

# %%