
FROM continuumio/miniconda3

MAINTAINER Eli Cutler <cutler.eli@gmail.com>

WORKDIR /nysparcs

COPY . .

RUN conda env create --file .config/environment.yaml

ENTRYPOINT ["conda", "run", "--name", "nysparcs"]

