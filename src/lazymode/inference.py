"""
Inference module for LazyMode model.

Provides standalone functions for loading and using the trained model
to format GitHub issues and pull requests.
"""

import os
from typing import Optional

from .model import LazyModeModel
from .data import generate_training_data, prepare_training_pairs


# Global model instance for convenience
_global_model: Optional[LazyModeModel] = None


def get_default_model_path() -> str:
    """Get the default path for the trained model."""
    return os.path.join(os.path.dirname(__file__), "..", "..", "models", "lazymode.pkl")


def load_model(
    filepath: Optional[str] = None,
    use_gpu: bool = True,
    force_retrain: bool = False
) -> LazyModeModel:
    """
    Load a trained LazyMode model.
    
    If no filepath is provided, uses the default model location.
    If the model doesn't exist, trains a new one.
    
    Args:
        filepath: Path to the model file. If None, uses default.
        use_gpu: Whether to use GPU if available.
        force_retrain: Force retraining even if model exists.
        
    Returns:
        Loaded or trained LazyModeModel.
    """
    global _global_model
    
    if filepath is None:
        filepath = get_default_model_path()
    
    # Check if we need to train a new model
    if not os.path.exists(filepath) or force_retrain:
        print(f"Model not found at {filepath}. Training new model...")
        model = train_and_save_model(filepath, use_gpu=use_gpu)
    else:
        model = LazyModeModel.load(filepath, use_gpu=use_gpu)
    
    _global_model = model
    return model


def train_and_save_model(
    filepath: str,
    use_gpu: bool = True
) -> LazyModeModel:
    """
    Train a new model and save it.
    
    Args:
        filepath: Path to save the model.
        use_gpu: Whether to use GPU if available.
        
    Returns:
        Trained LazyModeModel.
    """
    # Generate training data
    data = generate_training_data()
    pairs = prepare_training_pairs(data)
    inputs, outputs = zip(*pairs)
    
    # Create and train model
    model = LazyModeModel(n_neighbors=3, max_features=500, use_gpu=use_gpu)
    model.train(list(inputs), list(outputs))
    
    # Ensure directory exists
    dir_path = os.path.dirname(filepath) or "."
    os.makedirs(dir_path, exist_ok=True)
    
    # Save model
    model.save(filepath)
    
    return model


def format_github_issue(
    raw_input: str,
    model: Optional[LazyModeModel] = None,
    use_gpu: bool = True
) -> str:
    """
    Format a raw input into a GitHub-ready Markdown issue.
    
    This is the main standalone function for inference that can be
    easily copied into other projects.
    
    Args:
        raw_input: Raw text describing the issue (e.g., "App crashes on login").
        model: Optional pre-loaded model. If None, uses/loads global model.
        use_gpu: Whether to use GPU if available (only used if loading model).
        
    Returns:
        Formatted Markdown string ready for GitHub issue/PR.
        
    Example:
        >>> result = format_github_issue("Login button crashes the app")
        >>> print(result)
        ## Bug Report: Login Button Crashes The App
        ...
    """
    global _global_model
    
    if model is None:
        if _global_model is None:
            _global_model = load_model(use_gpu=use_gpu)
        model = _global_model
    
    return model.predict(raw_input)


def format_multiple_issues(
    raw_inputs: list,
    model: Optional[LazyModeModel] = None,
    use_gpu: bool = True
) -> list:
    """
    Format multiple raw inputs into GitHub-ready Markdown issues.
    
    Args:
        raw_inputs: List of raw input texts.
        model: Optional pre-loaded model.
        use_gpu: Whether to use GPU if available.
        
    Returns:
        List of formatted Markdown strings.
    """
    global _global_model
    
    if model is None:
        if _global_model is None:
            _global_model = load_model(use_gpu=use_gpu)
        model = _global_model
    
    return [model.predict(raw_input) for raw_input in raw_inputs]


def quick_format(raw_input: str) -> str:
    """
    Quick format function for one-off usage without model management.
    
    Creates an in-memory model, trains it, and formats the input.
    This is slower but doesn't leave any files on disk.
    
    Args:
        raw_input: Raw text describing the issue.
        
    Returns:
        Formatted Markdown string.
    """
    # Generate training data
    data = generate_training_data()
    pairs = prepare_training_pairs(data)
    inputs, outputs = zip(*pairs)
    
    # Create and train model in memory
    model = LazyModeModel(n_neighbors=3, max_features=500, use_gpu=False)
    model.train(list(inputs), list(outputs), verbose=False)
    
    return model.predict(raw_input)


# Alias for common use case
format_issue = format_github_issue


if __name__ == "__main__":
    # Demo usage
    test_inputs = [
        "App crashes on login button tap",
        "Database connection times out after 30 seconds",
        "Search feature returns no results",
        "Dark mode colors are wrong on settings page",
        "Push notifications not working on iOS",
    ]
    
    print("LazyMode - GitHub Issue Formatter Demo\n")
    print("=" * 60)
    
    for test_input in test_inputs:
        print(f"\n\nInput: {test_input}")
        print("-" * 60)
        result = format_github_issue(test_input, use_gpu=False)
        print(result)
        print("=" * 60)
