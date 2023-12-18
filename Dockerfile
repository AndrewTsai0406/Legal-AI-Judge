# FROM python:3.9.18-alpine3.18
FROM python:3.9.18-bookworm

RUN python -m pip install --upgrade pip
WORKDIR /app
COPY ["./requirements.txt", "./"]
RUN pip install -r requirements.txt

COPY ["./app-fastapi/.", "./"]
# RUN python load_model.py
COPY ["./app-fastapi/models/.", "./models/"]

# Make port 6969 available to the world outside this container
EXPOSE 6969

# Run gunicorn server when the container launches (for production)
# ENTRYPOINT ["gunicorn",'"app:app"', "-b=0.0.0.0:8000", "-k uvicorn.workers.UvicornWorker"]
CMD gunicorn -w 1 -b=0.0.0.0:8000 -k uvicorn.workers.UvicornWorker --timeout 900 app:app

# docker build -t legal-predict .
# docker run -it --rm -p 6969:8000 legal-predict
