FROM python:3.10

WORKDIR /app

COPY . .

RUN pip install fastapi uvicorn openenv-core openai

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]