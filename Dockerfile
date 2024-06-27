FROM python:3.8

WORKDIR /workspace/

COPY . /workspace/

RUN apt-get update

RUN pip3 install torch==1.10.1+cpu torchvision==0.11.2+cpu torchaudio==0.10.1 -f https://download.pytorch.org/whl/cpu/torch_stable.html
RUN pip3 install munch scikit-learn wandb scipy matplotlib==3.5.1 seaborn tabulate gym==0.25.2 pandas pytorch-lightning==1.7.2

ENTRYPOINT [ "/bin/bash" ]
