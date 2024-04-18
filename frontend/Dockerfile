FROM node:21.7.2-alpine
VOLUME /public/plots

RUN mkdir -p /public
WORKDIR /public

RUN npm i -g http-server

COPY client /public/client
COPY *.html /public/

EXPOSE 6060
CMD ["http-server", "-p", "6060"]
