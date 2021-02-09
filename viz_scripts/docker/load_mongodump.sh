MONGODUMP_FILE=$1

echo "Copying file to temporary location"
cp $1 /tmp
pushd /tmp

echo "Expanding archive"
tar xzf $1

echo "Checking dump file output"
ls -al | grep dump

echo "Copying dump to docker container"
docker cp dump em-public-dashboard_db_1:/tmp

echo "Restoring the dump"
docker exec em-public-dashboard_db_1 bash -c 'cd /tmp && mongorestore'
