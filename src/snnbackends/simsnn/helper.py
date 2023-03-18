"""Checks and asserts whether a list does not contain any duplicate
elements."""
from typing import List

from typeguard import typechecked


@typechecked
def contains_dupes(*, some_list: List) -> bool:
    """Returns true if a list contains duplicates, false otherwise."""
    return len(some_list) != len(set(some_list))


@typechecked
def assert_contains_no_dupes(*, some_list: List) -> None:
    """Raises ValueError if a list contains duplicates."""
    if contains_dupes(some_list=some_list):
        raise ValueError(f"Error, {some_list} contains dupes.")
