
FROM continuumio/miniconda3

MAINTAINER Eli Cutler <cutler.eli@gmail.com>

WORKDIR /nysparcs

CMD ["echo", "aaaay from $(pwd)"]

