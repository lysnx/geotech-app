from typing import Any, Union

def validate_positive(value: Union[int, float], name: str):
    """
    Validates that a value is positive.
    """
    if value <= 0:
        raise ValueError(f"{name} must be positive. Got {value}")

def validate_non_negative(value: Union[int, float], name: str):
    """
    Validates that a value is non-negative.
    """
    if value < 0:
        raise ValueError(f"{name} must be non-negative. Got {value}")

def validate_range(value: Union[int, float], min_val: float, max_val: float, name: str):
    """
    Validates that a value is within a range [min_val, max_val].
    """
    if not (min_val <= value <= max_val):
        raise ValueError(f"{name} must be between {min_val} and {max_val}. Got {value}")
