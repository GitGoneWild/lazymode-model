"""
Tests for LazyMode model.

This module contains tests for the data generation, model training,
and inference functionality.
"""

import os
import sys
import tempfile

import pytest

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from lazymode.data import (
    generate_training_data,
    load_dataset,
    prepare_training_pairs,
    save_dataset,
)
from lazymode.model import LazyModeModel, TextVectorizer
from lazymode.inference import format_github_issue, quick_format


class TestDataGeneration:
    """Tests for data generation module."""
    
    def test_generate_training_data_returns_list(self):
        """Test that generate_training_data returns a list."""
        data = generate_training_data()
        assert isinstance(data, list)
    
    def test_generate_training_data_has_minimum_examples(self):
        """Test that we generate at least 40 training examples."""
        data = generate_training_data()
        assert len(data) >= 40
    
    def test_training_data_has_correct_structure(self):
        """Test that each training example has input and output keys."""
        data = generate_training_data()
        for item in data:
            assert "input" in item
            assert "output" in item
            assert isinstance(item["input"], str)
            assert isinstance(item["output"], str)
    
    def test_training_data_outputs_are_markdown(self):
        """Test that outputs contain Markdown elements."""
        data = generate_training_data()
        for item in data:
            output = item["output"]
            # Should have headers
            assert "##" in output
            # Should have description section
            assert "Description" in output
            # Should have task items
            assert "- [ ]" in output
    
    def test_save_and_load_dataset(self):
        """Test saving and loading dataset."""
        data = generate_training_data()[:5]  # Use subset for speed
        
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            filepath = f.name
        
        try:
            save_dataset(data, filepath)
            loaded_data = load_dataset(filepath)
            
            assert len(loaded_data) == len(data)
            for original, loaded in zip(data, loaded_data):
                assert original["input"] == loaded["input"]
                assert original["output"] == loaded["output"]
        finally:
            os.unlink(filepath)
    
    def test_prepare_training_pairs(self):
        """Test converting dataset to training pairs."""
        data = [
            {"input": "test1", "output": "output1"},
            {"input": "test2", "output": "output2"},
        ]
        pairs = prepare_training_pairs(data)
        
        assert len(pairs) == 2
        assert pairs[0] == ("test1", "output1")
        assert pairs[1] == ("test2", "output2")


class TestTextVectorizer:
    """Tests for TextVectorizer class."""
    
    def test_vectorizer_tokenization(self):
        """Test tokenization of text."""
        vectorizer = TextVectorizer()
        tokens = vectorizer._tokenize("Hello World! This is a Test 123.")
        
        assert "hello" in tokens
        assert "world" in tokens
        assert "test" in tokens
        assert "123" in tokens
    
    def test_vectorizer_fit(self):
        """Test fitting vectorizer on texts."""
        vectorizer = TextVectorizer(max_features=10)
        texts = ["hello world", "hello python", "world python"]
        
        vectorizer.fit(texts)
        
        assert vectorizer.fitted
        assert len(vectorizer.vocabulary) <= 10
        assert "hello" in vectorizer.vocabulary
    
    def test_vectorizer_transform(self):
        """Test transforming texts to vectors."""
        vectorizer = TextVectorizer(max_features=10)
        texts = ["hello world", "hello python"]
        
        vectors = vectorizer.fit_transform(texts)
        
        assert vectors.shape[0] == 2
        assert vectors.shape[1] == len(vectorizer.vocabulary)
    
    def test_vectorizer_transform_without_fit_raises(self):
        """Test that transform without fit raises error."""
        vectorizer = TextVectorizer()
        
        with pytest.raises(ValueError):
            vectorizer.transform(["test"])


class TestLazyModeModel:
    """Tests for LazyModeModel class."""
    
    @pytest.fixture
    def trained_model(self):
        """Create a trained model for testing."""
        data = generate_training_data()[:20]  # Use subset for speed
        pairs = prepare_training_pairs(data)
        inputs, outputs = zip(*pairs)
        
        model = LazyModeModel(n_neighbors=3, max_features=200, use_gpu=False)
        model.train(list(inputs), list(outputs), verbose=False)
        return model
    
    def test_model_initialization(self):
        """Test model initialization."""
        model = LazyModeModel(n_neighbors=5, max_features=100, use_gpu=False)
        
        assert model.n_neighbors == 5
        assert model.max_features == 100
        assert model.device == "cpu"
        assert not model.is_trained
    
    def test_model_training(self, trained_model):
        """Test model training."""
        assert trained_model.is_trained
        assert len(trained_model.training_inputs) > 0
        assert len(trained_model.training_outputs) > 0
        assert trained_model.training_vectors is not None
    
    def test_model_training_mismatch_raises(self):
        """Test that mismatched inputs/outputs raises error."""
        model = LazyModeModel(use_gpu=False)
        
        with pytest.raises(ValueError):
            model.train(["input1", "input2"], ["output1"])
    
    def test_model_prediction_before_training_raises(self):
        """Test that prediction before training raises error."""
        model = LazyModeModel(use_gpu=False)
        
        with pytest.raises(ValueError):
            model.predict("test input")
    
    def test_model_prediction(self, trained_model):
        """Test model prediction."""
        result = trained_model.predict("Login button crashes the app")
        
        assert isinstance(result, str)
        assert "##" in result
        assert "Description" in result
    
    def test_model_prediction_has_required_sections(self, trained_model):
        """Test that predictions contain required sections."""
        result = trained_model.predict("Database connection timeout error")
        
        # Check for key Markdown sections
        assert "### Description" in result or "## Bug Report" in result
        assert "### Proposed Tasks" in result or "- [ ]" in result
    
    def test_model_prediction_performance(self, trained_model):
        """Test that prediction is fast (< 5 seconds)."""
        import time
        
        start = time.time()
        trained_model.predict("Test input for performance check")
        elapsed = time.time() - start
        
        assert elapsed < 5.0, f"Prediction took {elapsed:.2f}s, should be < 5s"
    
    def test_model_save_and_load(self, trained_model):
        """Test saving and loading model."""
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            filepath = f.name
        
        try:
            trained_model.save(filepath)
            
            loaded_model = LazyModeModel.load(filepath, use_gpu=False)
            
            assert loaded_model.is_trained
            assert loaded_model.n_neighbors == trained_model.n_neighbors
            
            # Test prediction with loaded model
            original_result = trained_model.predict("test input")
            loaded_result = loaded_model.predict("test input")
            
            assert original_result == loaded_result
        finally:
            os.unlink(filepath)
    
    def test_model_save_without_training_raises(self):
        """Test that saving untrained model raises error."""
        model = LazyModeModel(use_gpu=False)
        
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            temp_path = f.name
        
        try:
            with pytest.raises(ValueError):
                model.save(temp_path)
        finally:
            # Clean up if file was created (shouldn't happen since it raises)
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    def test_model_evaluate(self, trained_model):
        """Test model evaluation."""
        data = generate_training_data()[-5:]  # Use last 5 for validation
        pairs = prepare_training_pairs(data)
        inputs, outputs = zip(*pairs)
        
        metrics = trained_model.evaluate(list(inputs), list(outputs))
        
        assert "structural_accuracy" in metrics
        assert "avg_section_coverage" in metrics
        assert 0 <= metrics["structural_accuracy"] <= 1
        assert 0 <= metrics["avg_section_coverage"] <= 1
    
    def test_model_size_under_limit(self, trained_model):
        """Test that model size is under 500MB target."""
        size_bytes = trained_model.get_model_size()
        size_mb = size_bytes / (1024 * 1024)
        
        assert size_mb < 500, f"Model size {size_mb:.2f}MB exceeds 500MB limit"
    
    def test_extract_issue_type_bug(self):
        """Test issue type extraction for bugs."""
        model = LazyModeModel(use_gpu=False)
        
        assert model._extract_issue_type("app crashes on startup") == "Bug Report"
        assert model._extract_issue_type("login button not working") == "Bug Report"
        assert model._extract_issue_type("error when saving file") == "Bug Report"
    
    def test_extract_issue_type_feature(self):
        """Test issue type extraction for features."""
        model = LazyModeModel(use_gpu=False)
        
        assert model._extract_issue_type("add dark mode support") == "Feature Request"
        assert model._extract_issue_type("implement new login page") == "Feature Request"
    
    def test_extract_issue_type_performance(self):
        """Test issue type extraction for performance issues."""
        model = LazyModeModel(use_gpu=False)
        
        assert model._extract_issue_type("app is very slow") == "Performance Issue"
        assert model._extract_issue_type("memory usage too high") == "Performance Issue"


class TestInference:
    """Tests for inference module."""
    
    def test_format_github_issue(self):
        """Test format_github_issue function."""
        result = format_github_issue("Login page shows error", use_gpu=False)
        
        assert isinstance(result, str)
        assert "##" in result
    
    def test_format_github_issue_with_model(self):
        """Test format_github_issue with pre-loaded model."""
        data = generate_training_data()[:10]
        pairs = prepare_training_pairs(data)
        inputs, outputs = zip(*pairs)
        
        model = LazyModeModel(n_neighbors=3, use_gpu=False)
        model.train(list(inputs), list(outputs), verbose=False)
        
        result = format_github_issue("Database error", model=model)
        
        assert isinstance(result, str)
        assert "##" in result
    
    def test_quick_format(self):
        """Test quick_format function."""
        result = quick_format("Button click not responding")
        
        assert isinstance(result, str)
        assert "##" in result
        assert "Description" in result


class TestDiverseInputs:
    """Tests for diverse input handling (acceptance criteria)."""
    
    @pytest.fixture
    def model(self):
        """Create a trained model."""
        data = generate_training_data()
        pairs = prepare_training_pairs(data)
        inputs, outputs = zip(*pairs)
        
        model = LazyModeModel(n_neighbors=3, max_features=500, use_gpu=False)
        model.train(list(inputs), list(outputs), verbose=False)
        return model
    
    def test_diverse_input_1_crash(self, model):
        """Test crash-related input."""
        result = model.predict("App crashes when clicking submit button")
        
        assert "##" in result
        assert "Description" in result
        assert "- [ ]" in result
    
    def test_diverse_input_2_performance(self, model):
        """Test performance-related input."""
        result = model.predict("Website takes 20 seconds to load")
        
        assert "##" in result
        assert "Description" in result
    
    def test_diverse_input_3_feature(self, model):
        """Test feature request input."""
        result = model.predict("Add export to PDF functionality")
        
        assert "##" in result
        assert "Description" in result
    
    def test_diverse_input_4_ui(self, model):
        """Test UI-related input."""
        result = model.predict("Dark mode text is unreadable")
        
        assert "##" in result
        assert "Description" in result
    
    def test_diverse_input_5_api(self, model):
        """Test API-related input."""
        result = model.predict("REST API returns 500 internal server error")
        
        assert "##" in result
        assert "Description" in result
    
    def test_all_outputs_are_valid_markdown(self, model):
        """Test that all outputs are valid GitHub Markdown."""
        test_inputs = [
            "Login fails with wrong password",
            "Image upload not working",
            "Search returns empty results",
            "Session timeout too short",
            "Mobile app crashes on startup",
        ]
        
        for test_input in test_inputs:
            result = model.predict(test_input)
            
            # Basic Markdown structure checks
            assert result.startswith("##"), f"Output for '{test_input}' doesn't start with header"
            assert "###" in result, f"Output for '{test_input}' missing subsections"
            assert "- [ ]" in result, f"Output for '{test_input}' missing task checkboxes"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
