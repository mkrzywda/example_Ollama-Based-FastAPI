FROM ollama/ollama:latest
LABEL authors="Maciej Krzywda"

COPY startup.sh ./startup.sh

RUN chmod +x ./startup.sh

EXPOSE 11434

ENTRYPOINT ["/bin/sh"]
CMD ["./startup.sh"]