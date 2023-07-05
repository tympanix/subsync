FROM python:3.11-slim

RUN apt-get update && apt-get install ffmpeg -y

COPY . .
RUN pip install .

ENTRYPOINT [ "/usr/local/bin/subsync" ]
