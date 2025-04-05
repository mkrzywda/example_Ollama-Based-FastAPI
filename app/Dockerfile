FROM python:3.12-slim
LABEL authors="MaciejKrzywda"

WORKDIR /app

COPY . .

RUN pip install --upgrade pip && pip install -r requirements.txt

RUN chmod +x startup.sh

EXPOSE 8000
ENTRYPOINT ["/app/startup.sh"]
