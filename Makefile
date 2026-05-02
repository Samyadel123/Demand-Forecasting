.PHONY: build clean train

# Build the custom uv + java image
build:
	docker compose up --build

fast:
	docker compose up

# Run the cleaning pipeline
clean:
	docker compose run --rm pipeline_runner uv run python src/pipeline/cleaning_pipeline.py

# Run the training pipeline
train:
	docker compose run --rm pipeline_runner uv run python src/pipeline/ml_pipeline.py

# Check Hadoop/Spark configuration
check:
	docker compose run --rm pipeline_runner uv run python scripts/check_config.py
