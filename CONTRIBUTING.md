# Contributing to AI Village

We're excited that you're interested in contributing to AI Village! This document outlines the process for contributing to the project and provides guidelines for submitting pull requests.

## Table of Contents

1. [Setting Up the Development Environment](#setting-up-the-development-environment)
2. [Running Tests](#running-tests)
3. [Continuous Learning Features](#continuous-learning-features)
4. [Submitting Changes](#submitting-changes)
5. [Code Style Guidelines](#code-style-guidelines)
6. [Reporting Issues](#reporting-issues)

## Setting Up the Development Environment

1. Fork the repository on GitHub.
2. Clone your fork locally:
   ```
   git clone https://github.com/your-username/ai-village.git
   cd ai-village
   ```
3. Create a virtual environment and activate it:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```
4. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Running Tests

We use unittest for our testing framework. To run all tests:

```
python -m unittest discover -v
```

To run specific test files:

```
python -m unittest agents/king/tests/test_integration.py
```

Make sure all tests pass before submitting a pull request.

## Refreshing wheels

The CI pipeline installs dev dependencies from `vendor/wheels` without
internet access. Refresh the cache whenever `requirements-dev.txt` changes:

1. Edit `requirements-dev.txt` as needed.
2. Run `python scripts/fetch_wheels.py` to download wheels and update
   `docs/build_artifacts/wheel-manifest.txt`.
3. Commit **only** `vendor/wheels/.gitkeep` and the manifest file.
4. **Do not** commit any `.whl` artifacts.

If CI reports a wheel cache change, rerun the script and commit the updated
manifest.

## Continuous Learning Features

AI Village incorporates continuous learning capabilities. When working with these features:

1. Ensure that the `ContinuousLearner` class is properly integrated with the agent you're working on.
2. Test the learning process by providing diverse feedback and monitoring the agent's performance over time.
3. Verify that the learning rate adjusts appropriately based on the agent's performance.

## Submitting Changes

1. Create a new branch for your feature or bug fix:
   ```
   git checkout -b feature/your-feature-name
   ```
2. Make your changes and commit them with a clear commit message.
3. Push your changes to your fork:
   ```
   git push origin feature/your-feature-name
   ```
4. Create a pull request from your fork to the main AI Village repository.
5. Ensure that your pull request includes:
   - A clear description of the changes
   - Any relevant issue numbers
   - Updates to documentation if necessary

## Code Style Guidelines

- Follow PEP 8 guidelines for Python code style.
- Use type hints for function arguments and return values.
- Write clear, concise comments and docstrings.
- Use meaningful variable and function names.

## Reporting Issues

If you encounter a bug or have a suggestion for improvement:

1. Check if the issue already exists in the GitHub issue tracker.
2. If not, create a new issue with a clear title and description.
3. Include steps to reproduce the issue and any relevant error messages.

Thank you for contributing to AI Village! Your efforts help make this project better for everyone.
