FROM node:18-alpine
VOLUME /plots

RUN mkdir -p /public
WORKDIR /public

RUN npm install -g npm@8.14.0 && npm i -g http-server

COPY client /public/client
COPY *.html /public/

EXPOSE 6060
CMD ["http-server", "-p", "6060"]

