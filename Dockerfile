# syntax=docker/dockerfile:1

FROM python:3.8-slim-buster

WORKDIR /app

COPY requirements.txt requirements.txt
COPY . .
RUN pip install -e .
RUN pip install -r requirements.txt


CMD [ "python", "scripts/download_model.py" , "--bart_large_only" ]
CMD [ "python", "scripts/chatbot.py" ]