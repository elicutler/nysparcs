
FROM continuumio/miniconda3

MAINTAINER Eli Cutler <cutler.eli@gmail.com>

WORKDIR /nysparcs

# Create conda env before copying other files so that 
# rebuild can use cached conda env even if other files change
COPY .config/environment.yaml .config/environment.yaml
RUN conda env create --file .config/environment.yaml

COPY . .

ENTRYPOINT ["conda", "run", "--name", "nysparcs"]
