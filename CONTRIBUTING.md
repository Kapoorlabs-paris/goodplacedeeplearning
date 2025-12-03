# Contributing to Keras Deep Learning Course

Thank you for your interest in contributing to this course! Here are some guidelines to help you get started.

## How to Contribute

### 1. Report Issues

If you find bugs, typos, or unclear explanations:
- Open an issue with a clear description
- Include relevant error messages and code snippets
- Specify which notebook and module the issue is in

### 2. Improve Notebooks

To enhance existing notebooks:
1. Fork the repository
2. Create a branch: `git checkout -b improve/notebook-name`
3. Make your changes
4. Test the notebook end-to-end
5. Create a pull request with a clear description

### 3. Add New Content

To contribute new notebooks or modules:
1. Follow the existing naming convention: `NN_descriptive_name.ipynb`
2. Include clear explanations and comments
3. Add visualizations where appropriate
4. Test with fresh kernel start
5. Update the README with new notebook information

## Notebook Guidelines

### Structure
- Clear title and description at the top
- Markdown cells explaining concepts before code
- Code cells with comments for complex operations
- Visualizations and results clearly presented

### Code Style
```python
# Use clear variable names
learning_rate = 0.001  # Better than lr = 0.001

# Add comments for complex logic
# Normalize features to zero mean and unit variance
X_normalized = (X - X.mean()) / X.std()

# Use meaningful function names
def train_model(X_train, y_train, epochs=10):
    """Train a deep learning model."""
    pass
```

### Requirements
- Run the entire notebook from top to bottom
- Clear any outputs before committing
- Add markdown explanations for each section
- Include learning objectives at the beginning
- Add a summary or key takeaways at the end

## Development Setup

```bash
# Install development dependencies
pip install -e ".[dev]"

# Format code with black
black Keras/

# Check code style with flake8
flake8 Keras/

# Run tests
pytest tests/
```

## Code of Conduct

- Be respectful and inclusive
- Provide constructive feedback
- Help others learn
- No harassment or discrimination

## Review Process

1. All contributions go through peer review
2. Maintainers will check:
   - Accuracy of content
   - Code quality
   - Notebook execution
   - Compliance with guidelines
3. You may be asked to make revisions
4. Once approved, your contribution will be merged

## Questions?

- Check existing issues and discussions first
- Open a discussion for questions
- Be clear and provide context

## License

By contributing, you agree your work will be licensed under the same license as the project.

Thank you for helping make this course better! 🚀
