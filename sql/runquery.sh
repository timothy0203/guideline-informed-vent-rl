#!/bin/bash

# DBNAME="mimic_demo"
# # DBNAME="mimic"
# SCHEMA_QRY="mimiciii"
# SCHEMA="mimiciii"

# if [ -z $1 ]; then
# 	echo "Run script with query: ./runquery.sh queryfile.sql"
# fi

# sed "s/$SCHEMA_QRY/$SCHEMA/g" $1  | psql $DBNAME

# DBNAME="mimic_demo"
DBNAME="mimic"
SCHEMA_QRY="mimiciii"
SCHEMA="mimiciii"
USERNAME="postgres"  # Specify the username here

if [ -z $1 ]; then
	echo "Run script with query: ./runquery.sh queryfile.sql"
fi

sed "s/$SCHEMA_QRY/$SCHEMA/g" $1 | psql -U $USERNAME $DBNAME