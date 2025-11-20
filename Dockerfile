# Use an official Python runtime as a parent image
FROM python:3.12-slim

# Set the working directory in the container
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y build-essential curl && rm -rf /var/lib/apt/lists/*

# Install Poetry
RUN curl -sSL https://install.python-poetry.org | python3 -

# Add Poetry to the system PATH
ENV PATH="/root/.local/bin:$PATH"

# Configure Poetry to create the virtualenv in the project's directory
RUN poetry config virtualenvs.in-project true

# Copy dependency definition files
COPY poetry.lock pyproject.toml ./

# Install project dependencies
RUN poetry install --no-root --without dev

# Add the virtualenv's bin directory to the PATH
ENV PATH="/app/.venv/bin:$PATH"

# --- Pre-download and cache the embedding model to a shared location ---
# Create a global cache directory for Hugging Face models
RUN mkdir /cache
# Set the environment variable to use this directory (the modern approach)
ENV HF_HOME=/cache
# Run the download as root, which will place the model in /cache
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# Copy the application source code
COPY src/ ./src/

# Copy the static UI files
COPY static/ ./static/

# --- Create a non-root user for security ---
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Change ownership of all application-related directories
RUN chown -R appuser:appuser /app /cache

# Switch to the non-root user
USER appuser

# The port the app will run on
EXPOSE 8080

# Command to run the application
CMD ["uvicorn", "src.baler.main:app", "--host", "0.0.0.0", "--port", "8080"]
