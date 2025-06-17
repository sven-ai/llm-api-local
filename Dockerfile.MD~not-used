FROM python:3.13

# # RUN pip install -U -r py.reqs.list
# # RUN apt-get update && apt-get install -y iputils-ping

# # RUN mkdir /sven
# COPY /src /sven
# WORKDIR /sven

# ENTRYPOINT ["python", "server.py"]


RUN mkdir /sven
COPY run.sh /sven

ENTRYPOINT ["run.sh"]