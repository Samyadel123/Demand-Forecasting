.PHONY: build clean train

# Build the custom uv + java image
build:
	docker compose up --build

fast:
	docker compose up

# Run the cleaning pipeline
clean:
	docker compose run --rm pipeline_runner uv run python src/pipeline/cleaning_pipeline.py

# Run the training pipeline (includes feature engineering)
train:
	docker compose run --rm pipeline_runner uv run python src/pipeline/training_pipeline.py

# Run the full end-to-end pipeline (cleaning -> training -> serve)
pipeline: clean train serve

# Start the FastAPI inference server
serve:
	docker compose up api

# Check Hadoop/Spark configuration
check:
	docker compose run --rm pipeline_runner uv run python scripts/check_config.py
