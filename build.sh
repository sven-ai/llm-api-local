#!/usr/bin/env bash

# docker login --username=svenai

IMAGE_NAME="svenai/llm-api:local"

docker buildx use rc-builder

docker buildx build --platform=linux/amd64,linux/arm64 \
	-f "Dockerfile" \
	--push -t $IMAGE_NAME .
	 # > /dev/null
	# --output=type=registry \

# echo "OK. Built: $IMAGE_NAME"

# 
# Apple Silicon
# I found the platform: linux/arm64/v8 works properly.
# 