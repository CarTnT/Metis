#!/bin/bash

export PYTHONPATH=/data/Metis:$PYTHONPATH && python /data/Metis/app/controller/manage.py runserver 0.0.0.0:9001
#uwsgi --ini /data/Metis/app/controller/uwsgi.ini
#tail -f /dev/null

