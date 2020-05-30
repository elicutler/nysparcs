
FROM continuumio/miniconda3

MAINTAINER Eli Cutler <cutler.eli@gmail.com>

WORKDIR /nysparcs

COPY . .

EXPOSE 80

CMD ["ls", "-l", "-a"]

