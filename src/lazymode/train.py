"""
Training script for LazyMode model.

This script handles the end-to-end training process including:
- Data generation/loading
- Model training
- Model evaluation
- Model saving
"""

import argparse
import os
import sys
import time
from typing import List, Tuple

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from lazymode.data import generate_training_data, prepare_training_pairs, save_dataset
from lazymode.model import LazyModeModel


def split_data(
    pairs: List[Tuple[str, str]], train_ratio: float = 0.8
) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]:
    """
    Split data into training and validation sets.

    Args:
        pairs: List of (input, output) pairs.
        train_ratio: Ratio of data for training.

    Returns:
        Training pairs and validation pairs.
    """
    split_idx = int(len(pairs) * train_ratio)
    return pairs[:split_idx], pairs[split_idx:]


def train_model(
    use_gpu: bool = True,
    save_path: str = "models/lazymode.pkl",
    data_path: str = "data/training_data.json",
    verbose: bool = True,
) -> LazyModeModel:
    """
    Train the LazyMode model.

    Args:
        use_gpu: Whether to use GPU if available.
        save_path: Path to save the trained model.
        data_path: Path to save/load training data.
        verbose: Whether to print progress.

    Returns:
        Trained model.
    """
    start_time = time.time()

    if verbose:
        print("=" * 60)
        print("LazyMode Model Training")
        print("=" * 60)
        print()

    # Generate training data
    if verbose:
        print("Generating training data...")

    data = generate_training_data()
    save_dataset(data, data_path)

    if verbose:
        print(f"Generated {len(data)} training examples")
        print(f"Training data saved to {data_path}")
        print()

    # Prepare data
    pairs = prepare_training_pairs(data)
    train_pairs, val_pairs = split_data(pairs, train_ratio=0.9)

    train_inputs, train_outputs = zip(*train_pairs)
    val_inputs, val_outputs = zip(*val_pairs)

    if verbose:
        print(f"Training examples: {len(train_inputs)}")
        print(f"Validation examples: {len(val_inputs)}")
        print()

    # Create and train model
    if verbose:
        print("Initializing model...")

    model = LazyModeModel(n_neighbors=3, max_features=500, use_gpu=use_gpu)

    if verbose:
        print("Training model...")

    metrics = model.train(list(train_inputs), list(train_outputs), verbose=verbose)

    # Evaluate model
    if verbose:
        print()
        print("Evaluating model...")

    eval_metrics = model.evaluate(list(val_inputs), list(val_outputs))

    if verbose:
        print(f"Structural accuracy: {eval_metrics['structural_accuracy']:.2%}")
        print(f"Section coverage: {eval_metrics['avg_section_coverage']:.2%}")
        print()

    # Save model
    if verbose:
        print(f"Saving model to {save_path}...")

    model.save(save_path)

    # Print summary
    elapsed_time = time.time() - start_time
    model_size_mb = model.get_model_size() / (1024 * 1024)

    if verbose:
        print()
        print("=" * 60)
        print("Training Complete!")
        print("=" * 60)
        print(f"Training time: {elapsed_time:.2f} seconds")
        print(f"Model size (in memory): {model_size_mb:.2f} MB")
        print(f"Device used: {model.device}")
        print(f"Vocabulary size: {metrics['vocabulary_size']}")
        print()

    return model


def run_demo(model: LazyModeModel) -> None:
    """
    Run a demo of the trained model.

    Args:
        model: Trained model.
    """
    print()
    print("=" * 60)
    print("Model Demo")
    print("=" * 60)

    test_inputs = [
        "Application freezes when loading dashboard",
        "API returns 500 error on user registration",
        "Mobile app battery drain issue",
        "Need to add dark mode support",
        "Login session expires too quickly",
    ]

    for test_input in test_inputs:
        print()
        print(f"Input: {test_input}")
        print("-" * 60)
        start = time.time()
        result = model.predict(test_input)
        inference_time = time.time() - start
        print(result[:500] + "..." if len(result) > 500 else result)
        print(f"\n[Inference time: {inference_time * 1000:.2f} ms]")
        print("=" * 60)


def main():
    """Main entry point for training script."""
    parser = argparse.ArgumentParser(
        description="Train the LazyMode model for GitHub issue formatting"
    )
    parser.add_argument("--no-gpu", action="store_true", help="Disable GPU usage (use CPU only)")
    parser.add_argument(
        "--model-path",
        type=str,
        default="models/lazymode.pkl",
        help="Path to save the trained model",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default="data/training_data.json",
        help="Path to save training data",
    )
    parser.add_argument("--demo", action="store_true", help="Run demo after training")
    parser.add_argument("--quiet", action="store_true", help="Reduce output verbosity")

    args = parser.parse_args()

    model = train_model(
        use_gpu=not args.no_gpu,
        save_path=args.model_path,
        data_path=args.data_path,
        verbose=not args.quiet,
    )

    if args.demo:
        run_demo(model)


if __name__ == "__main__":
    main()
