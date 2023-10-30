#!/bin/bash

DBNAME="mimic"
SCHEMA_QRY="mimiciii"
SCHEMA="mimiciii"

if [ -z $1 ]; then
	echo "Run script with query: ./runquery.sh queryfile.sql"
fi

sed "s/$SCHEMA_QRY/$SCHEMA/g" $1  | psql $DBNAME
