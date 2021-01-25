FROM pytorch/pytorch:latest

RUN echo export PATH="$HOME/.local/bin:$PATH"

RUN mkdir /app

RUN apt-get update
RUN apt-get -y install wget
RUN apt-get install -y git build-essential \
  gcc make yasm autoconf curl \
  automake cmake libtool \
  checkinstall libmp3lame-dev \
  pkg-config libunwind-dev \
  zlib1g-dev libssl-dev

RUN python -m spacy download en
RUN pip install --upgrade pip
ADD requirements.txt /app/
RUN pip install -r /app/requirements.txt
COPY . /app
WORKDIR /app/

RUN export LC_ALL="en_US.UTF-8"
RUN export LC_CTYPE="en_US.UTF-8"

ENTRYPOINT ["tail", "-f", "/dev/null"]

CMD ["start"]
