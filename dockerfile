# Use a base image
FROM python:3.11

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements.txt file to the container
COPY requirement.txt .

# Install the Python dependencies
RUN pip install --no-cache-dir -r requirement.txt
RUN apt-get update

# Copy the application files to the container
COPY search.py .

# Run the Flask application
CMD ["python", "autotag_flask.py"]
