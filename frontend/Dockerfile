FROM node:18-alpine
VOLUME /plots

RUN mkdir -p /public
WORKDIR /public

RUN npm install -g npm@8.14.0 && npm i http-server

COPY client /public/client
COPY *.html /public/

EXPOSE 8080
CMD ["http-server"]

