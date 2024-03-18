from __future__ import annotations

import os

from langchain_openai import AzureChatOpenAI
from langchain.globals import set_debug, set_verbose
from sqlalchemy.engine import URL


MODEL_NAME = "gpt-3.5-turbo"
CHUNK_SIZE = int(250)
CHUNK_OVERLAP = int(0)
# Max concurrency for the model.
MAX_CONCURRENCY = 1

set_debug(True)
set_verbose(True)


def get_postgres_url() -> URL:
    url = URL.create(
        drivername="postgresql",
        username="langchain",
        password="langchain",
        host=os.environ.get("PG_HOST", "localhost"),
        database="langchain",
        port=5432,
    )
    return url


def get_model() -> AzureChatOpenAI:
    """Get the model."""
    return AzureChatOpenAI(
        deployment_name="gpt-35-turbo-0125",
        model_version="0125",
        temperature=0,
    )
