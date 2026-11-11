
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy requirements
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Expose Streamlit port
EXPOSE 8501

# Run Streamlit application
CMD ["streamlit", "run", "app/app.py", "--server.address=0.0.0.0", "--server.port=8201"]
