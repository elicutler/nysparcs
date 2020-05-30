
FROM continuumio/miniconda3

MAINTAINER Eli Cutler <cutler.eli@gmail.com>

WORKDIR /nysparcs

COPY . .

RUN conda env create -f .config/environment.yaml

EXPOSE 80

ENTRYPOINT ["conda", "run", "--name", "nysparcs"]

