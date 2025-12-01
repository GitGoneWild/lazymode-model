"""
LazyMode - Lightweight AI Model for GitHub Issue/PR Markdown Formatting

This package provides a lightweight AI model that transforms raw user input
into polished Markdown formatted specifically for GitHub issues or pull requests.
"""

__version__ = "0.1.0"

from .model import LazyModeModel
from .inference import format_github_issue, load_model
from .data import generate_training_data, load_dataset

__all__ = [
    "LazyModeModel",
    "format_github_issue",
    "load_model",
    "generate_training_data",
    "load_dataset",
]
