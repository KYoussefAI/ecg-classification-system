FROM python:3.12-slim
WORKDIR /workspace
COPY backend/requirements.txt /tmp/requirements.txt
COPY scripts/install_cpu.py /tmp/install_cpu.py
RUN python /tmp/install_cpu.py --requirements /tmp/requirements.txt
COPY training ./training
COPY backend ./backend
RUN useradd --create-home appuser && mkdir /data && chown appuser /data
USER appuser
EXPOSE 8000
CMD ["python", "-m", "uvicorn", "app.main:app", "--app-dir", "backend", "--host", "0.0.0.0", "--port", "8000"]
