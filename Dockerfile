FROM python:2-alpine
ENV PYTHONUNBUFFERED 1
ADD . /code/
RUN pip install -r /code/requirements.txt