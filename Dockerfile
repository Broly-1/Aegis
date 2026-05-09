FROM python:3.10-slim

# Hugging Face Spaces require running as a non-root user
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

WORKDIR /app

# Copy requirements and install dependencies
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY --chown=user . .

# Hugging Face Spaces route web traffic to port 7860
EXPOSE 7860

# Start the FastAPI server on port 7860
CMD ["uvicorn", "api.server:app", "--host", "0.0.0.0", "--port", "7860"]
