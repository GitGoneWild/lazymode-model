# LazyMode - Lightweight AI Model for GitHub Issue/PR Markdown Formatting

LazyMode is a lightweight AI model that transforms raw user input into polished Markdown formatted specifically for GitHub issues or pull requests. It ensures the output follows standard GitHub issue templates with all required sections.

## Features

- 🚀 **Lightweight & Fast** - Runs on CPU with minimal RAM usage (< 500MB model size)
- 🎯 **GPU Auto-Detection** - Automatically uses GPU if available for improved performance
- ⚡ **Quick Training** - Trains on ~50 examples in seconds
- 📝 **Structured Output** - Generates Markdown with title, description, environment, steps, expected/actual behavior, logs, and tasks
- 🔧 **Easy Integration** - Standalone function for use in scripts, notebooks, or applications

## Installation

```bash
# Clone the repository
git clone https://github.com/GitGoneWild/lazymode-model.git
cd lazymode-model

# Install dependencies
pip install -e .

# Or install just the requirements
pip install -r requirements.txt
```

## Quick Start

### Using the Standalone Function

```python
from lazymode import format_github_issue

# Format a raw input into a GitHub issue
result = format_github_issue("App crashes on login button tap")
print(result)
```

### Training Your Own Model

```python
from lazymode import LazyModeModel, generate_training_data, prepare_training_pairs

# Generate training data
data = generate_training_data()
pairs = prepare_training_pairs(data)
inputs, outputs = zip(*pairs)

# Create and train the model
model = LazyModeModel(n_neighbors=3, use_gpu=True)
model.train(list(inputs), list(outputs))

# Use the model
result = model.predict("Database connection timeout error")
print(result)

# Save for later use
model.save("models/lazymode.pkl")
```

### Using the Command Line

```bash
# Train the model
python -m lazymode.train --demo

# Or with options
python -m lazymode.train --no-gpu --model-path my_model.pkl
```

## Example Output

**Input:** `"Login button crashes the app"`

**Output:**
```markdown
## Bug Report: Login Button Crashes The App

### Description
Login button crashes the app

### Environment
- **Platform**: Mobile Application
- **Component**: Authentication/Login

### Steps to Reproduce
1. Open the application
2. Navigate to the login screen
3. Enter credentials (any valid or invalid)
4. Tap the login button
5. Observe the crash

### Expected Behavior
The application should process the login attempt and either authenticate the user or display an error message.

### Actual Behavior
The application crashes immediately upon tapping the login button.

### Error Logs
```
// Add crash logs here
```

### Proposed Tasks
- [ ] Investigate crash logs to identify root cause
- [ ] Add null checks for login button handler
- [ ] Implement proper error handling
- [ ] Add unit tests for login functionality
- [ ] Test fix on all supported platforms
```

## Project Structure

```
lazymode-model/
├── src/
│   └── lazymode/
│       ├── __init__.py      # Package exports
│       ├── data.py          # Training data generation
│       ├── model.py         # LazyModeModel implementation
│       ├── inference.py     # Inference utilities
│       └── train.py         # Training script
├── tests/
│   └── test_lazymode.py     # Unit tests
├── notebooks/
│   └── lazymode_demo.ipynb  # Interactive demo notebook
├── data/                    # Training data storage
├── models/                  # Saved model storage
├── requirements.txt         # Dependencies
├── setup.py                 # Package setup
└── README.md               # This file
```

## Requirements

- Python 3.8+
- NumPy >= 1.21.0
- (Optional) PyTorch >= 2.0.0 for GPU acceleration

## Running Tests

```bash
# Install test dependencies
pip install pytest pytest-cov

# Run tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=lazymode --cov-report=term-missing
```

## Jupyter Notebook Demo

Open `notebooks/lazymode_demo.ipynb` for an interactive demonstration of all features.

## API Reference

### `LazyModeModel`

```python
class LazyModeModel:
    def __init__(self, n_neighbors=3, max_features=500, use_gpu=True):
        """Initialize the model."""
        
    def train(self, inputs, outputs, verbose=True):
        """Train the model on input-output pairs."""
        
    def predict(self, input_text):
        """Generate formatted Markdown output."""
        
    def save(self, filepath):
        """Save the trained model."""
        
    @classmethod
    def load(cls, filepath, use_gpu=True):
        """Load a trained model."""
        
    def evaluate(self, test_inputs, test_outputs):
        """Evaluate model performance."""
```

### `format_github_issue`

```python
def format_github_issue(raw_input, model=None, use_gpu=True):
    """
    Format a raw input into a GitHub-ready Markdown issue.
    
    Args:
        raw_input: Raw text describing the issue.
        model: Optional pre-loaded model.
        use_gpu: Whether to use GPU if available.
        
    Returns:
        Formatted Markdown string.
    """
```

## License

MIT License

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
