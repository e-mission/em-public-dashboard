MONGODUMP_FILE=$1

echo "Copying file to docker container"
docker cp MONGODUMP_FILE em-public-dashboard_db_1:/tmp

echo "Restoring the dump"
docker exec em-public-dashboard_db_1 bash -c 'cd /tmp && tar xvf $MONGODUMP_FILE && mongorestore'
