FROM nvcr.io/nvidia/pytorch:20.12-py3

# Installing packages and deleting auxiliary files to reduce image size
RUN apt-get -y update \
 && DEBIAN_FRONTEND=noninteractive apt-get -y install sudo curl locales libboost-all-dev python-dev libhunspell-dev\
 && rm -rf /var/lib/apt/lists/*

RUN locale-gen en_US.UTF-8
ENV LANG en_US.UTF-8
ENV LANGUAGE en_US:en
ENV LC_ALL en_US.UTF-8

# Directories to be created within the image
RUN mkdir /models
RUN mkdir /app
RUN mkdir /experiments
RUN mkdir /corpora

# ‘docker build’ parameters
ARG USER_ID=1000
ARG GROUP_ID=1000

# Create the user 'user' in the image with the user id and group id specified in the parameters:
RUN addgroup --gid $GROUP_ID user
RUN adduser --disabled-password --gecos '' --uid $USER_ID --gid $GROUP_ID user
RUN echo "user ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/90-user

# Switch to new user
USER user
ENV HOME=/home/user

RUN chmod 777 /home/user

ENV PATH="/home/user/.local/bin:${PATH}"

# Install the sacrebleu branch that allows to use spBLEU
WORKDIR /app
RUN git clone --single-branch --branch adding_spm_tokenized_bleu https://github.com/ngoyal2707/sacrebleu.git >/dev/null 2>&1
WORKDIR /app/sacrebleu
RUN python setup.py install
WORKDIR /home/user


# Create conda environment for bicleaner-ai and install bicleaner-ai
RUN conda create -n bicleaner-ai python=3.8.5
RUN echo "conda activate bicleaner-ai" >> ~/.bashrc
RUN pip install bicleaner-ai
RUN pip install tensorflow==2.4.0
ENV LD_LIBRARY_PATH /usr/local/cuda-11.1/lib64
RUN sudo ln -s /usr/local/cuda-11.1/targets/x86_64-linux/lib/libcusolver.so.11 /usr/local/cuda-11.1/targets/x86_64-linux/lib/libcusolver.so.10

#fix lm_stats bug
#write_metadata(args, classifier, y_true, y_pred) #, lm_stats)
#/home/user/.local/lib/python3.8/site-packages/bicleaner_ai/bicleaner_ai_train.py
RUN sed -i 's/write_metadata(args, classifier, y_true, y_pred, lm_stats)/write_metadata(args, classifier, y_true, y_pred) #, lm_stats)/' /home/user/.local/lib/python3.8/site-packages/bicleaner_ai/bicleaner_ai_train.py
RUN sed -i 's/def write_metadata(args, classifier, y_true, y_pred, lm_stats):/def write_metadata(args, classifier, y_true, y_pred): #, lm_stats):/' /home/user/.local/lib/python3.8/site-packages/bicleaner_ai/training.py
RUN sed -i '285,293d' /home/user/.local/lib/python3.8/site-packages/bicleaner_ai/training.py
RUN sed -i '289,296d' /home/user/.local/lib/python3.8/site-packages/bicleaner_ai/training.py

RUN echo "conda deactivate bicleaner-ai" >> ~/.bashrc

WORKDIR /home/user
RUN git clone https://github.com/kpu/kenlm
WORKDIR /home/user/kenlm
RUN pip install . --install-option="--max_order 7"
RUN mkdir -p build
WORKDIR /home/user/kenlm/build
RUN cmake .. -DKENLM_MAX_ORDER=7 -DCMAKE_INSTALL_PREFIX:PATH=/home/user/kenlm
RUN make -j all install
WORKDIR /home/user


# Install dependencies
RUN pip install fairseq==0.10.2 sacrebleu==1.5.1 regex==2020.11.13 fastBPE==0.1.0 sacremoses==0.0.43 subword_nmt==0.3.7 sentencepiece==0.1.94 fairscale==0.3.1 pyidaungsu

# Fix mBART50 n-1 error for Fairseq 0.10.2 (version-dependent fix): https://github.com/pytorch/fairseq/issues/3474#issuecomment-832585357
# /home/user/.local/lib/python3.8/site-packages/fairseq/checkpoint_utils.py
# state["args"].arch="mbart_large"
RUN grep -c denoising_large /home/user/.local/lib/python3.8/site-packages/fairseq/checkpoint_utils.py || sed -i '435 i \ \ \ \ if state["args"].arch=="denoising_large":\n        state["args"].arch="mbart_large"' /home/user/.local/lib/python3.8/site-packages/fairseq/checkpoint_utils.py

WORKDIR /app

CMD ["nvidia-smi"]
