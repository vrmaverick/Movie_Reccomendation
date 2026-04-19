# Official Python 3.11 slim image
FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy everything (including the src folder and CSV)
COPY . .

# Expose port 80 as required 
EXPOSE 80

# Use 'src.main:app' and point to port 80 [cite: 6, 10]
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "80"]