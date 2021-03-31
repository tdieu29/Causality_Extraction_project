FROM python:3.7

# Copy all local files to the folder /app
COPY . /app
WORKDIR /app

# Upgrade pip and install the requirements.txt file
RUN pip install -U pip
RUN pip install -r requirements.txt

ENV PYTHONPATH="/app"

# Expose port you want your app on
EXPOSE 8501

# Run
ENTRYPOINT ["streamlit", "run", "CausalityExtraction/streamlit.py"]