# python 3
FROM emission/e-mission-server:4.0.0

ADD environment36.dashboard.additions.yml /

RUN /bin/bash -c "cd e-mission-server && cat setup/export_versions.sh && printf \"export EXP_CONDA_VER=4.12.0\" > setup/export_versions.sh && cat setup/export_versions.sh"
RUN /bin/bash -c "cd e-mission-server && cat setup/environment36.notebook.additions.yml && printf \"name: emission\nchannels:\n- conda-forge\n- defaults\ndependencies:\n- branca=0.5.0\n- folium=0.12.1.post1\n- ipython=7.33.0\n- jedi=0.18.1\n- jupyter=1.0.0\n- matplotlib=3.5.1\n\" > setup/environment36.notebook.additions.yml && cat setup/environment36.notebook.additions.yml" 
RUN /bin/bash -c "cd e-mission-server && cat setup/export_versions.sh && source setup/setup_conda.sh Linux-x86_64 && source setup/setup.sh"
RUN /bin/bash -c "cd e-mission-server && source setup/activate.sh && conda env update --name emission --file setup/environment36.notebook.additions.yml"
RUN /bin/bash -c "cd e-mission-server && source setup/activate.sh && conda list && conda env update --name emission --file /environment36.dashboard.additions.yml"

ADD start_notebook.sh /usr/src/app/start_notebook.sh
RUN chmod u+x /usr/src/app/start_notebook.sh

ADD crontab /usr/src/app/crontab

EXPOSE 8888

CMD ["/bin/bash", "/usr/src/app/start_notebook.sh"]
