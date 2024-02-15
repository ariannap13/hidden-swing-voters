from huggingface/transformers-pytorch-gpu

ENV APP_ROOT=/app
RUN mkdir -p ${APP_ROOT} && chown 1001:0 ${APP_ROOT} && chmod 775 ${APP_ROOT} ; \
    useradd -d ${APP_ROOT} -g 0 -u 1001 -M -s /bin/bash process
WORKDIR ${APP_ROOT}

COPY requirements.txt .
RUN python3 -m pip install --no-cache-dir -r requirements.txt

COPY *.py label_dict-golden.pkl  defaults.cfg ./
COPY i3/ ./i3
USER 1001

CMD ["uvicorn"  , "--port", "8000", "--host", "0.0.0.0", "pt_api:app"]