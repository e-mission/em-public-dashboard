FROM node:25.6.0-alpine3.23
VOLUME /public/plots

RUN mkdir -p /public
WORKDIR /public

RUN npm install -g npm@latest
RUN npm i -g http-server

COPY client /public/client
COPY *.html /public/

EXPOSE 6060
CMD ["http-server", "-p", "6060"]
