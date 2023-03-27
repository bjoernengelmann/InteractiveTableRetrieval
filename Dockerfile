FROM python:3.8-slim


RUN apt-get update && apt-get install -y \
	git \
	openjdk-17-jdk \
	openjdk-17-jre \
	postgresql-client 


RUN mkdir /workspace

RUN pip install --upgrade pip
RUN pip install --upgrade git+https://github.com/terrier-org/pyterrier.git#egg=python-terrier \
                          streamlit
                          
RUN pip install -U pip setuptools wheel
RUN pip install -U spacy
RUN python -m spacy download en_core_web_sm
RUN pip install 'spacy[transformers]'
RUN pip install seaborn

RUN python -m spacy download en_core_web_trf

RUN groupadd user && useradd -r -m -g user user
USER user
