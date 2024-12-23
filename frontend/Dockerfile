FROM node:21.7.2-alpine
VOLUME /public/plots

RUN mkdir -p /public
WORKDIR /public

RUN npm i -g http-server

COPY client /public/client
COPY *.html /public/

# expose directive does not do anything, but for documentation purposes,
# the port 3274 is mapped in the docker-compose.yml.
EXPOSE 3274

CMD ["http-server", "-p", "3274"]
