# %%

MAIN = "__main__" == __name__

from typing import Union

# %%

def divide(a: Union[float, int], b: Union[float, int]) -> Union[float, None]:
    try:
        result = a / b
    except ZeroDivisionError:
        print("Error: Cannot divide by zero.")
        return None
    except TypeError:
        print("Error: Inputs must be numbers.")
        return None
    except Exception as e:
        print(f"Unexpected error: {e}")
        raise # Re-raise the exception for any other unexpected errors
    else: # Execute if no exception occurred
        return result
    finally: # Always executes
        print("Division attempt finished.")

# Test cases
if MAIN:
    print(divide(10, 2))  # Should return 5.0
    print(divide(10, 0))  # Should handle division by zero
    print(divide(10, "a"))  # Should handle invalid input
    print(divide(10, None))  # Should handle invalid input

# %%