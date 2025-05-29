FROM python:3.9.12

RUN mkdir -p /home/app

WORKDIR /home/app

COPY . .

RUN python -m pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt
RUN apt-get update
RUN apt-get install libglu1 -y
RUN apt-get install libxcursor-dev -y
RUN apt-get install libxinerama1 -y

RUN apt-get install -y x11-apps \
    && rm -rf /var/lib/apt/lists/* 

ENV DISPLAY=:0

EXPOSE 8080

CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8080", "--no-browser", "--allow-root", "--NotebookApp.token=''"]
