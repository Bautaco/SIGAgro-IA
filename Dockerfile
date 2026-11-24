# SIGAgro IA - imagen para despliegue (Linux, CPU)
# El modelo se descarga al construir la imagen: en ejecución no necesita internet.
FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    HF_HOME=/opt/hf \
    MODELO_ID=wambugu71/crop_leaf_diseases_vit \
    MODELO_REVISION=7d5b32bcd6f83a2f57e7e0346358fad276296877

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu

# Descarga anticipada del modelo (queda en la imagen)
RUN python -c "import os; from transformers import AutoImageProcessor, AutoModelForImageClassification as M; \
i=os.environ['MODELO_ID']; r=os.environ['MODELO_REVISION']; AutoImageProcessor.from_pretrained(i, revision=r); M.from_pretrained(i, revision=r)"

COPY app ./app
ENV HF_HUB_OFFLINE=1
EXPOSE 8000
CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
