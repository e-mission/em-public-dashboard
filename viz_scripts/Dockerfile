# python 3
ARG SERVER_IMAGE_TAG
FROM shankari/e-mission-server:master_${SERVER_IMAGE_TAG}

ADD https://s3.amazonaws.com/rds-downloads/rds-combined-ca-bundle.pem /etc/ssl/certs/

VOLUME /plots

ADD environment.yml /

WORKDIR /usr/src/app

RUN /bin/bash -c "source setup/activate.sh && conda env update --name emission --file /environment.yml"

RUN mkdir -p /usr/src/app/saved-notebooks
WORKDIR /usr/src/app/saved-notebooks

COPY bin ./bin
COPY *.ipynb .
COPY *.py .

# Delete all test packages since they generate false positives in the vulnerability scan
# e.g.
# root/miniconda-4.12.0/pkgs/conda-4.12.0-py38h06a4308_0/info/test/tests/data/env_metadata/py27-osx-no-binary/lib/python2.7/site-packages/requests-2.19.1-py2.7.egg-info/PKG-INFO
# RUN rm -rf /root/miniconda-26.1.1/pkgs/jupyterlab-4.4.6-pyhd8ed1ab_0/site-packages/jupyterlab/staging
# RUN rm -rf /root/miniconda-26.1.1/pkgs/jupyterlab-4.4.6-pyhd8ed1ab_0/site-packages/jupyterlab/tests
# RUN rm -rf /root/miniconda-26.1.1/envs/emission/lib/python3.11/site-packages/jupyterlab/tests
# RUN rm -rf /root/miniconda-26.1.1/lib/python3.12/site-packages/jupyterlab/tests

# These are from the conda cache
RUN /bin/bash -c "find /root/miniconda-*/pkgs -wholename \*info/test\* -type d | xargs rm -rf"
RUN /bin/bash -c "find /root/miniconda-*/pkgs -wholename \*staging\* -type d | xargs rm -rf"
RUN /bin/bash -c "find /root/miniconda-*/pkgs -wholename \*tests\* -type d | xargs rm -rf"

# These are from the installed versions
RUN /bin/bash -c "find /root/miniconda-*/lib/python*/site-packages -wholename \*info/test\* -type d | xargs rm -rf"
RUN /bin/bash -c "find /root/miniconda-*/lib/python*/site-packages -wholename \*staging\* -type d | xargs rm -rf"
RUN /bin/bash -c "find /root/miniconda-*/lib/python*/site-packages -wholename \*tests\* -type d | xargs rm -rf"

# And this is from our environment
RUN /bin/bash -c "find /root/miniconda-*/envs/emission -wholename \*info/test\* -type d | xargs rm -rf"
RUN /bin/bash -c "find /root/miniconda-*/envs/emission -wholename \*staging\* -type d | xargs rm -rf"
RUN /bin/bash -c "find /root/miniconda-*/envs/emission -wholename \*tests\* -type d | xargs rm -rf"

WORKDIR /usr/src/app

ADD docker/start_notebook.sh /usr/src/app/.docker/start_notebook.sh
RUN chmod u+x /usr/src/app/.docker/start_notebook.sh

ADD docker/crontab /usr/src/app/crontab

EXPOSE 8888

CMD ["/bin/bash", "/usr/src/app/.docker/start_notebook.sh"]
