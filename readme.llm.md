This is a python-based project.

It has a pluggable structure. Loaded code is called modules, loading handled by `loader.py`. Usually there is a `<modulename>.yml` config file that defines a module to load via `active` param value - the value is a name of the `<modulename>.py` file to be loaded as module.

## Important Notes

- **FastAPI `BackgroundTasks` does NOT work with gunicorn `uvicorn_worker.UvicornWorker`** - tasks scheduled via `BackgroundTasks.add_task()` may not execute reliably when running under gunicorn with uvicorn workers. Use thread pool executors or `asyncio.create_task()` directly for background work instead.

