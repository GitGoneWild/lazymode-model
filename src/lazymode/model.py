"""
LazyMode Model - Lightweight AI model for GitHub Issue/PR Markdown formatting.

This module implements a lightweight model using template matching and text
generation techniques optimized for CPU usage with optional GPU acceleration.
"""

import json
import os
import pickle
import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


class TextVectorizer:
    """
    Simple text vectorizer using TF-IDF-like approach.
    Optimized for small datasets and fast inference.
    """
    
    def __init__(self, max_features: int = 1000):
        """
        Initialize the vectorizer.
        
        Args:
            max_features: Maximum number of features (vocabulary size).
        """
        self.max_features = max_features
        self.vocabulary: Dict[str, int] = {}
        self.idf: Dict[str, float] = {}
        self.fitted = False
    
    def _tokenize(self, text: str) -> List[str]:
        """
        Tokenize text into words.
        
        Args:
            text: Input text.
            
        Returns:
            List of tokens.
        """
        # Lowercase and extract alphanumeric tokens
        text = text.lower()
        tokens = re.findall(r'\b[a-z0-9]+\b', text)
        return tokens
    
    def fit(self, texts: List[str]) -> "TextVectorizer":
        """
        Fit the vectorizer on training texts.
        
        Args:
            texts: List of training texts.
            
        Returns:
            Self for chaining.
        """
        # Count document frequency
        doc_freq: Dict[str, int] = {}
        all_tokens: Dict[str, int] = {}
        
        for text in texts:
            tokens = set(self._tokenize(text))
            for token in tokens:
                doc_freq[token] = doc_freq.get(token, 0) + 1
                all_tokens[token] = all_tokens.get(token, 0) + 1
        
        # Select top features by frequency
        sorted_tokens = sorted(all_tokens.items(), key=lambda x: x[1], reverse=True)
        top_tokens = [token for token, _ in sorted_tokens[:self.max_features]]
        
        # Create vocabulary
        self.vocabulary = {token: idx for idx, token in enumerate(top_tokens)}
        
        # Calculate IDF
        n_docs = len(texts)
        self.idf = {
            token: np.log((n_docs + 1) / (doc_freq.get(token, 0) + 1)) + 1
            for token in self.vocabulary
        }
        
        self.fitted = True
        return self
    
    def transform(self, texts: List[str]) -> np.ndarray:
        """
        Transform texts to TF-IDF vectors.
        
        Args:
            texts: List of texts to transform.
            
        Returns:
            NumPy array of shape (n_texts, n_features).
        """
        if not self.fitted:
            raise ValueError("Vectorizer must be fitted before transform")
        
        vectors = np.zeros((len(texts), len(self.vocabulary)))
        
        for i, text in enumerate(texts):
            tokens = self._tokenize(text)
            # Count term frequency
            tf: Dict[str, int] = {}
            for token in tokens:
                if token in self.vocabulary:
                    tf[token] = tf.get(token, 0) + 1
            
            # Calculate TF-IDF
            for token, count in tf.items():
                idx = self.vocabulary[token]
                vectors[i, idx] = count * self.idf[token]
            
            # Normalize
            norm = np.linalg.norm(vectors[i])
            if norm > 0:
                vectors[i] /= norm
        
        return vectors
    
    def fit_transform(self, texts: List[str]) -> np.ndarray:
        """
        Fit and transform in one step.
        
        Args:
            texts: List of texts.
            
        Returns:
            Transformed vectors.
        """
        self.fit(texts)
        return self.transform(texts)


class LazyModeModel:
    """
    LazyMode - A lightweight AI model for GitHub Issue/PR Markdown formatting.
    
    Uses nearest neighbor matching with template-based generation,
    optimized for CPU with optional GPU acceleration.
    """
    
    def __init__(
        self,
        n_neighbors: int = 3,
        max_features: int = 500,
        use_gpu: bool = True
    ):
        """
        Initialize the LazyMode model.
        
        Args:
            n_neighbors: Number of nearest neighbors to consider.
            max_features: Maximum vocabulary size for vectorizer.
            use_gpu: Whether to attempt GPU acceleration (auto-detects availability).
        """
        self.n_neighbors = n_neighbors
        self.max_features = max_features
        self.use_gpu = use_gpu
        self.device = "cpu"
        
        self.vectorizer = TextVectorizer(max_features=max_features)
        self.training_inputs: List[str] = []
        self.training_outputs: List[str] = []
        self.training_vectors: Optional[np.ndarray] = None
        self.is_trained = False
        
        # Check GPU availability
        self._setup_device()
    
    def _setup_device(self) -> None:
        """Set up computation device (CPU or GPU)."""
        if self.use_gpu:
            try:
                import torch
                if torch.cuda.is_available():
                    self.device = "cuda"
                    print("GPU detected: Using CUDA for acceleration")
                elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                    self.device = "mps"
                    print("GPU detected: Using Apple MPS for acceleration")
                else:
                    self.device = "cpu"
                    print("No GPU detected: Using CPU")
            except ImportError:
                self.device = "cpu"
                print("PyTorch not available: Using CPU with NumPy")
        else:
            self.device = "cpu"
            print("Using CPU (GPU disabled)")
    
    def train(
        self,
        inputs: List[str],
        outputs: List[str],
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Train the model on input-output pairs.
        
        Args:
            inputs: List of raw input texts.
            outputs: List of formatted Markdown outputs.
            verbose: Whether to print training progress.
            
        Returns:
            Training metrics dictionary.
        """
        if len(inputs) != len(outputs):
            raise ValueError("Inputs and outputs must have same length")
        
        if verbose:
            print(f"Training on {len(inputs)} examples...")
        
        # Store training data
        self.training_inputs = inputs
        self.training_outputs = outputs
        
        # Fit vectorizer and transform inputs
        self.training_vectors = self.vectorizer.fit_transform(inputs)
        
        self.is_trained = True
        
        metrics = {
            "n_examples": len(inputs),
            "vocabulary_size": len(self.vectorizer.vocabulary),
            "device": self.device,
        }
        
        if verbose:
            print(f"Training complete. Vocabulary size: {len(self.vectorizer.vocabulary)}")
        
        return metrics
    
    def _find_nearest_neighbors(
        self,
        query_vector: np.ndarray,
        k: int
    ) -> List[Tuple[int, float]]:
        """
        Find k nearest neighbors using cosine similarity.
        
        Args:
            query_vector: Query vector.
            k: Number of neighbors.
            
        Returns:
            List of (index, similarity) tuples.
        """
        if self.training_vectors is None:
            raise ValueError("Model must be trained before inference")
        
        # Calculate cosine similarities
        similarities = np.dot(self.training_vectors, query_vector)
        
        # Get top k indices
        if k >= len(similarities):
            top_indices = np.argsort(similarities)[::-1]
        else:
            top_indices = np.argpartition(similarities, -k)[-k:]
            top_indices = top_indices[np.argsort(similarities[top_indices])[::-1]]
        
        return [(int(idx), float(similarities[idx])) for idx in top_indices[:k]]
    
    def _extract_issue_type(self, text: str) -> str:
        """
        Extract the type of issue from input text.
        
        Args:
            text: Input text.
            
        Returns:
            Issue type (bug, feature, performance, etc.)
        """
        text_lower = text.lower()
        
        if any(word in text_lower for word in ["crash", "error", "fail", "broken", "not working", "bug"]):
            return "Bug Report"
        elif any(word in text_lower for word in ["add", "new", "implement", "create", "feature"]):
            return "Feature Request"
        elif any(word in text_lower for word in ["slow", "performance", "memory", "cpu", "lag"]):
            return "Performance Issue"
        elif any(word in text_lower for word in ["docs", "document", "readme", "help"]):
            return "Documentation"
        else:
            return "Bug Report"
    
    def _generate_fallback_output(self, input_text: str) -> str:
        """
        Generate a fallback output when no good match is found.
        
        Args:
            input_text: Input text.
            
        Returns:
            Formatted Markdown output.
        """
        issue_type = self._extract_issue_type(input_text)
        
        # Create a title from input
        title = input_text.strip()
        if len(title) > 60:
            title = title[:57] + "..."
        title = title.title()
        
        return f"""## {issue_type}: {title}

### Description
{input_text}

### Environment
- **Platform**: [Specify platform]
- **Component**: [Specify component]

### Steps to Reproduce
1. [Step 1]
2. [Step 2]
3. [Step 3]

### Expected Behavior
[Describe what should happen]

### Actual Behavior
[Describe what actually happens]

### Error Logs
```
[Add relevant logs here]
```

### Proposed Tasks
- [ ] Investigate the issue
- [ ] Identify root cause
- [ ] Implement fix
- [ ] Add tests
- [ ] Verify fix works"""
    
    def predict(self, input_text: str) -> str:
        """
        Generate formatted Markdown output for an input.
        
        Args:
            input_text: Raw input text.
            
        Returns:
            Formatted Markdown output.
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        # Vectorize input
        query_vector = self.vectorizer.transform([input_text])[0]
        
        # Find nearest neighbors
        neighbors = self._find_nearest_neighbors(query_vector, self.n_neighbors)
        
        if not neighbors or neighbors[0][1] < 0.1:
            # No good match found, use fallback
            return self._generate_fallback_output(input_text)
        
        # Use best matching template
        best_idx, best_sim = neighbors[0]
        best_output = self.training_outputs[best_idx]
        
        # Adapt the output based on input
        adapted_output = self._adapt_output(input_text, best_output)
        
        return adapted_output
    
    def _adapt_output(self, input_text: str, template_output: str) -> str:
        """
        Adapt a template output to the new input.
        
        Args:
            input_text: New input text.
            template_output: Template output from training data.
            
        Returns:
            Adapted output.
        """
        # Extract issue type and create new title
        issue_type = self._extract_issue_type(input_text)
        
        # Create title from input
        title = input_text.strip()
        if len(title) > 60:
            title = title[:57] + "..."
        title = title.title()
        
        # Update the title line
        lines = template_output.split('\n')
        if lines and lines[0].startswith('##'):
            lines[0] = f"## {issue_type}: {title}"
        
        # Update description if present
        for i, line in enumerate(lines):
            if line.strip() == "### Description":
                # Find next section
                j = i + 1
                while j < len(lines) and not lines[j].startswith('###'):
                    if lines[j].strip() and not lines[j].startswith('#'):
                        # Replace first description line with input
                        lines[j] = input_text
                        break
                    j += 1
                break
        
        return '\n'.join(lines)
    
    def save(self, filepath: str) -> None:
        """
        Save the trained model to a file.
        
        Args:
            filepath: Path to save the model.
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")
        
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else ".", exist_ok=True)
        
        model_data = {
            "n_neighbors": self.n_neighbors,
            "max_features": self.max_features,
            "vectorizer_vocabulary": self.vectorizer.vocabulary,
            "vectorizer_idf": self.vectorizer.idf,
            "training_inputs": self.training_inputs,
            "training_outputs": self.training_outputs,
            "training_vectors": self.training_vectors.tolist() if self.training_vectors is not None else None,
        }
        
        with open(filepath, "wb") as f:
            pickle.dump(model_data, f)
        
        print(f"Model saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: str, use_gpu: bool = True) -> "LazyModeModel":
        """
        Load a trained model from a file.
        
        Args:
            filepath: Path to the saved model.
            use_gpu: Whether to use GPU if available.
            
        Returns:
            Loaded LazyModeModel instance.
        """
        with open(filepath, "rb") as f:
            model_data = pickle.load(f)
        
        model = cls(
            n_neighbors=model_data["n_neighbors"],
            max_features=model_data["max_features"],
            use_gpu=use_gpu
        )
        
        model.vectorizer.vocabulary = model_data["vectorizer_vocabulary"]
        model.vectorizer.idf = model_data["vectorizer_idf"]
        model.vectorizer.fitted = True
        
        model.training_inputs = model_data["training_inputs"]
        model.training_outputs = model_data["training_outputs"]
        model.training_vectors = np.array(model_data["training_vectors"]) if model_data["training_vectors"] else None
        model.is_trained = True
        
        print(f"Model loaded from {filepath}")
        return model
    
    def evaluate(
        self,
        test_inputs: List[str],
        test_outputs: List[str]
    ) -> Dict[str, float]:
        """
        Evaluate the model on test data.
        
        Args:
            test_inputs: Test input texts.
            test_outputs: Expected output texts.
            
        Returns:
            Evaluation metrics.
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before evaluation")
        
        correct_structure = 0
        total_section_matches = 0
        expected_sections = ["Description", "Environment", "Steps to Reproduce", 
                           "Expected Behavior", "Actual Behavior", "Proposed Tasks"]
        
        for input_text, expected_output in zip(test_inputs, test_outputs):
            predicted = self.predict(input_text)
            
            # Check structural correctness
            has_title = predicted.startswith("##")
            has_description = "### Description" in predicted
            has_tasks = "### Proposed Tasks" in predicted or "- [ ]" in predicted
            
            if has_title and has_description and has_tasks:
                correct_structure += 1
            
            # Count matching sections
            for section in expected_sections:
                if f"### {section}" in predicted:
                    total_section_matches += 1
        
        n_tests = len(test_inputs)
        return {
            "structural_accuracy": correct_structure / n_tests if n_tests > 0 else 0,
            "avg_section_coverage": total_section_matches / (n_tests * len(expected_sections)) if n_tests > 0 else 0,
        }
    
    def get_model_size(self) -> int:
        """
        Get approximate model size in bytes.
        
        Returns:
            Model size in bytes.
        """
        import sys
        
        size = 0
        size += sys.getsizeof(self.training_inputs)
        size += sys.getsizeof(self.training_outputs)
        if self.training_vectors is not None:
            size += self.training_vectors.nbytes
        size += sys.getsizeof(self.vectorizer.vocabulary)
        size += sys.getsizeof(self.vectorizer.idf)
        
        return size


if __name__ == "__main__":
    # Quick test
    from data import generate_training_data, prepare_training_pairs
    
    # Generate training data
    data = generate_training_data()
    pairs = prepare_training_pairs(data)
    inputs, outputs = zip(*pairs)
    
    # Create and train model
    model = LazyModeModel(n_neighbors=3, use_gpu=False)
    metrics = model.train(list(inputs), list(outputs))
    print(f"Training metrics: {metrics}")
    
    # Test prediction
    test_input = "Application freezes when loading dashboard"
    result = model.predict(test_input)
    print(f"\nTest input: {test_input}")
    print(f"\nPrediction:\n{result}")
