# Contributing to LazyMode

Thank you for your interest in contributing to LazyMode! This document provides guidelines and instructions for contributing.

## Getting Started

### Prerequisites

- Python 3.8 or higher
- pip package manager
- Git

### Development Setup

1. Fork and clone the repository:
   ```bash
   git clone https://github.com/your-username/lazymode-model.git
   cd lazymode-model
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install development dependencies:
   ```bash
   pip install -e ".[dev]"
   ```

4. Verify your setup by running tests:
   ```bash
   pytest tests/ -v
   ```

## Development Workflow

### Code Style

We use [Ruff](https://docs.astral.sh/ruff/) for linting and formatting. Before submitting changes:

```bash
# Check for linting issues
ruff check src/ tests/

# Auto-fix linting issues
ruff check --fix src/ tests/

# Format code
ruff format src/ tests/
```

### Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=lazymode --cov-report=term-missing

# Run a specific test file
pytest tests/test_lazymode.py -v

# Run a specific test
pytest tests/test_lazymode.py::TestLazyModeModel::test_model_prediction -v
```

### Pre-commit Hooks (Optional)

We recommend using pre-commit hooks to catch issues before committing:

```bash
pip install pre-commit
pre-commit install
```

## Making Changes

### Branch Naming

Use descriptive branch names:
- `feature/add-new-template` for new features
- `fix/login-crash-bug` for bug fixes
- `docs/update-readme` for documentation changes
- `refactor/improve-vectorizer` for refactoring

### Commit Messages

Follow the [Conventional Commits](https://www.conventionalcommits.org/) specification:

```
type(scope): description

[optional body]

[optional footer]
```

Examples:
- `feat(model): add support for feature request templates`
- `fix(inference): handle empty input strings`
- `docs(readme): add installation troubleshooting section`
- `test(model): add edge case tests for prediction`

### Pull Request Process

1. Create a new branch from `main`
2. Make your changes with appropriate tests
3. Ensure all tests pass and code is formatted
4. Update documentation if needed
5. Submit a pull request with a clear description

## Code Guidelines

### Python Style

- Follow PEP 8 guidelines
- Use type hints for function parameters and return values
- Write docstrings for all public functions and classes
- Keep functions focused and small (single responsibility)

### Testing

- Write tests for new features and bug fixes
- Maintain or improve code coverage
- Test edge cases and error conditions
- Use descriptive test names

### Documentation

- Update README.md for user-facing changes
- Add docstrings to new functions and classes
- Include examples in docstrings when helpful

## Reporting Issues

### Bug Reports

Include:
- Python version and OS
- Steps to reproduce
- Expected vs actual behavior
- Error messages and tracebacks

### Feature Requests

Include:
- Use case description
- Proposed solution
- Alternative approaches considered

## Questions?

Feel free to open an issue for questions or discussions about the project.

---

Thank you for contributing! 🎉
