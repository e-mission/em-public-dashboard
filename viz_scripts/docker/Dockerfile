# python 3
FROM emission/e-mission-server:2.9.2

RUN /bin/bash -c "cd e-mission-server && source setup/activate.sh && conda env update --name emission --file setup/environment36.notebook.additions.yml"
RUN /bin/bash -c "cd e-mission-server && source setup/activate.sh && pip install nbparameterise"

ADD start_notebook.sh /usr/src/app/start_notebook.sh
RUN chmod u+x /usr/src/app/start_notebook.sh

EXPOSE 8888

CMD ["/bin/bash", "/usr/src/app/start_notebook.sh"]
