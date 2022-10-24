FROM continuumio/anaconda3

# Install from conda env
COPY environment.yml /tmp/environment.yml

# Conda packages
RUN conda env create -f /tmp/environment.yml -n mipcat

EXPOSE 8888

#CMD cd ${MAIN_PATH} && sh run_jupyterlab.sh
CMD /bin/bash
