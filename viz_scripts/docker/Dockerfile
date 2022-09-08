# python 3
FROM emission/e-mission-server.dev.server-only:4.0.0
ARG SERVER_REPO=https://github.com/aGuttman/e-mission-server.git
ARG SERVER_BRANCH=dashboard-dependencies

VOLUME /plots

ADD docker/environment36.dashboard.additions.yml /

RUN /bin/bash -c "/clone_server.sh"

WORKDIR /usr/src/app

RUN /bin/bash -c "cd e-mission-server && source setup/activate.sh && conda env update --name emission --file setup/environment36.notebook.additions.yml"
RUN /bin/bash -c "cd e-mission-server && source setup/activate.sh && conda env update --name emission --file /environment36.dashboard.additions.yml"

RUN mkdir -p /usr/src/app/saved-notebooks
WORKDIR /usr/src/app/saved-notebooks

COPY auxiliary_files .
COPY bin .
COPY conf .
COPY *.ipynb .
COPY *.py .

WORKDIR /usr/src/app

ADD docker/start_notebook.sh /usr/src/app/start_notebook.sh
RUN chmod u+x /usr/src/app/start_notebook.sh

ADD docker/crontab /usr/src/app/crontab

EXPOSE 8888

CMD ["/bin/bash", "/usr/src/app/start_notebook.sh"]
