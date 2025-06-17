#!/usr/bin/env bash
#
# dev worflow
#

if [ -z "$ANTHROPIC_API_KEY" ]; then
   echo "ERR. ANTHROPIC_API_KEY env expected"
   echo "Set it like so in a Terminal: export ANTHROPIC_API_KEY=your_api_key_here"
   echo "Get your key at https://console.anthropic.com"

   exit 1
fi


docker stop sven-llm-api; docker rm sven-llm-api;

docker run --rm \
	-e ANTHROPIC_API_KEY="$ANTHROPIC_API_KEY" \
	--network=sven --hostname=sven-llm-api -p 12345:12345 \
	-w /sven -v $PWD:/sven \
	-v $HOME/sven/llm-api-data:/data -v ~/.cache:/root/.cache \
	--name=sven-llm-api \
	python:3.13 /bin/sh -c 'pip install -r py.reqs.list && gunicorn --access-logfile - --log-level info -w 4 -k uvicorn.workers.UvicornWorker --chdir /sven/src server:app --bind 0.0.0.0:12345'
