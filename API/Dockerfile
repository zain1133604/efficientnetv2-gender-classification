# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Set TORCH_HOME to a writable directory within the container
# This is crucial for PyTorch to cache pre-trained models
ENV TORCH_HOME=/app/torch_cache

# Create the directory for the torch cache
RUN mkdir -p ${TORCH_HOME}

# Copy the requirements file into the container at /app
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code and model weights into the container at /app
# This includes main.py and best_modeln.pth
COPY . .

# Make port 8000 available to the world outside this container
EXPOSE 8000

# Run the Uvicorn server when the container launches
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]