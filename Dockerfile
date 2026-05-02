FROM python:3.12-slim-bookworm

#1. Install java, uv, and system dependencies for LightGBM/XGBoost
RUN apt-get update && apt-get install -y --no-install-recommends \
    openjdk-17-jre-headless \
    curl \
    ca-certificates \
    procps \
    libgomp1 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# 3. Set up the working directory
WORKDIR /app

# 4. Install dependencies
# We copy these first to leverage Docker's cache layer
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project

# 5. Set environment variables for Spark and Java
ENV JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
ENV PATH="/app/.venv/bin:$PATH"

# 6. Copy the rest of your application code
COPY . .

# 7. Final sync to install the 'resynor' project itself
RUN uv sync --frozen

# running the app 

