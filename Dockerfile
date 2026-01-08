# 1. Use the specific Python version you developed on
FROM python:3.10.12-slim

# 2. Set the working directory inside the container
WORKDIR /app

# 3. Copy the requirements file first (for caching)
COPY requirements.txt .

# 4. Install dependencies (including Streamlit)
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy the rest of your app's code and models
COPY . .

# 6. Expose the port Hugging Face expects (7860)
EXPOSE 7860

# 7. The command to start the app
CMD ["streamlit", "run", "app.py", "--server.port=7860", "--server.address=0.0.0.0"]