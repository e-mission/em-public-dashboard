# python 3
ARG SERVER_IMAGE_TAG
FROM ghcr.io/e-mission/e-mission-server:${SERVER_IMAGE_TAG}
LABEL org.opencontainers.image.source=https://github.com/e-mission/em-public-dashboard

VOLUME /plots

ADD docker/environment36.dashboard.additions.yml /

WORKDIR /usr/src/app

RUN /bin/bash -c "source setup/activate.sh && conda env update --name emission --file setup/environment36.notebook.additions.yml"
RUN /bin/bash -c "source setup/activate.sh && conda env update --name emission --file /environment36.dashboard.additions.yml"

RUN mkdir -p /usr/src/app/saved-notebooks
WORKDIR /usr/src/app/saved-notebooks

COPY auxiliary_files ./auxiliary_files
COPY bin ./bin
COPY *.ipynb .
COPY *.py .

# Delete all test packages since they generate false positives in the vulnerability scan
# e.g.
# root/miniconda-4.12.0/pkgs/conda-4.12.0-py38h06a4308_0/info/test/tests/data/env_metadata/py27-osx-no-binary/lib/python2.7/site-packages/requests-2.19.1-py2.7.egg-info/PKG-INFO
# root/miniconda-4.12.0/pkgs/conda-4.12.0-py38h06a4308_0/info/test/tests/data/env_metadata/py36-osx-whl/lib/python3.6/site-packages/Django-2.1.dist-info/METADATA
# root/miniconda-4.12.0/pkgs/conda-4.12.0-py38h06a4308_0/info/test/tests/data/env_metadata/py36-osx-whl/lib/python3.6/site-packages/Scrapy-1.5.1.dist-info/METADATA

RUN /bin/bash -c "find /root/miniconda-*/pkgs -wholename \*info/test\* -type d | xargs rm -rf"

WORKDIR /usr/src/app

ADD docker/start_notebook.sh /usr/src/app/.docker/start_notebook.sh
RUN chmod u+x /usr/src/app/.docker/start_notebook.sh

ADD docker/crontab /usr/src/app/crontab

EXPOSE 8888

CMD ["/bin/bash", "/usr/src/app/.docker/start_notebook.sh"]
