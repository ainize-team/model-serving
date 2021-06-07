FROM pytorch/pytorch:1.7.1-cuda11.0-cudnn8-runtime
RUN mkdir workspace
WORKDIR /workspace
ADD requirements.txt .
COPY . .
RUN pip install -r requirements.txt
EXPOSE 5000
CMD ["python","index.py"]