FROM tensorflow/tensorflow:latest-py3

ARG TL_VERSION

RUN echo "Container Tag: ${TF_CONTAINER_VERSION}" \
    && apt-get update \
    && case $TF_CONTAINER_VERSION in \
            latest-py3 | latest-gpu-py3) apt-get install -y python3-tk  ;; \
            *)                           apt-get install -y python-tk ;; \
        esac \
    && if [ -z "$TL_VERSION" ]; then \
        echo "Building a Nightly Release" \
        && apt-get install -y git \
        && mkdir /dist/ && cd /dist/ \
        && git clone https://github.com/tensorlayer/tensorlayer.git \
        && cd tensorlayer \
        && pip install --disable-pip-version-check --no-cache-dir --upgrade -e .[all]; \
    else \
        echo "Building Tag Release: $TL_VERSION" \
        && pip install  --disable-pip-version-check --no-cache-dir --upgrade tensorlayer[all]=="$TL_VERSION"; \
    fi \
    && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/*

#Environment Variables for Click
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

COPY requirements.txt /tmp/
RUN pip3 install --requirement /tmp/requirements.txt

COPY . /chatbot

WORKDIR /chatbot
RUN python3 main.py --batch-size 32 --num-epochs 1 -lr 0.001

CMD python3 main.py --inference-mode
