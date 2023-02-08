FROM ubuntu:latest

# Create app directory
WORKDIR /app

RUN apt-get -y update && apt-get -y install python3 && apt-get -y install python3-pip

# Install app dependencies
COPY requirements.txt ./

RUN pip install --upgrade pip

RUN pip install -r requirements.txt

# Bundle app source
COPY . .

EXPOSE 5000
CMD ["flask", "run","--host","0.0.0.0","--port","5000"]
