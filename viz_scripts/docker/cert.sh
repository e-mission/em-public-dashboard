if [ -z ${PROD_STAGE} ] ; then
    echo “Not in staging / production environment, continuing build...” 
elif [ ${PROD_STAGE} = "TRUE" ] ; then
    wget https://s3.amazonaws.com/rds-downloads/rds-combined-ca-bundle.pem -O /etc/ssl/certs/rds-combined-ca-bundle.pem
    echo "In staging / production environment, added AWS certificates"
fi
