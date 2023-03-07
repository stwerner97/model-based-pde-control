FROM python:3.8


WORKDIR /workspace/

RUN apt-get update

RUN pip3 install torch==1.10.1+cpu torchvision==0.11.2+cpu torchaudio==0.10.1 -f https://download.pytorch.org/whl/cpu/torch_stable.html
RUN pip3 install munch
RUN pip3 install sklearn
RUN pip3 install wandb
RUN pip3 install scipy
RUN pip3 install matplotlib==3.5.1
RUN pip3 install seaborn
RUN pip3 install tabulate
RUN pip3 install gym==0.25.2
RUN pip3 install pandas
RUN pip3 install pytorch-lightning==1.7.2


ENTRYPOINT [ "/bin/bash" ]
