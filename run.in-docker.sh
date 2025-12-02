pip install -r py.reqs.list

playwright install-deps
playwright install

gunicorn --access-logfile - --log-level info -w 64 --timeout 180 -k uvicorn.workers.UvicornWorker --chdir /sven/src server:app --bind 0.0.0.0:12345
