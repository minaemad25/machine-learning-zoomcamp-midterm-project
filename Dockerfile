FROM python:3.12-slim

WORKDIR /app

# install dependencies first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# copy project code and data
COPY train.py predict.py test_sample.json ./
COPY data/company_bankruptcy.csv ./data/

# train the final model during the image build so no pickle has to be committed
RUN python train.py

EXPOSE 9696

HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:9696/ping')" || exit 1

ENTRYPOINT ["gunicorn", "--bind=0.0.0.0:9696", "predict:app"]
