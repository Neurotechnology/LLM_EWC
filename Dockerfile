FROM nvcr.io/nvidia/pytorch:24.10-py3
WORKDIR /LLM_EWC
COPY requirements.txt .
RUN pip install -r requirements.txt
