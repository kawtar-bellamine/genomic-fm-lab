FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch \
 && pip install --no-cache-dir -r requirements.txt
COPY . .
RUN pytest -q
# full study: pretraining, fine-tuning grid, benchmarks, figures (about 1.5 h on 2 CPU cores)
CMD ["python", "run.py"]
