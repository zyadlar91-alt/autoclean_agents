.PHONY: install data inject run eval clean notebook help

## variables
PYTHON = venv/bin/python3
DATA_RAW = data/raw/sncf_tgv_raw.csv
DATA_NOISY = data/raw/sncf_tgv_noisy.csv
DATA_CLEANED = data/cleaned/sncf_tgv_cleaned.csv

install:
	$(PYTHON) -m pip install -r requirements.txt
## NB: i proceeded by cretaion a virtual environment

data:
	mkdir -p data/raw
	curl -L "https://ressources.data.sncf.com/api/explore/v2.1/catalog/datasets/regularite-mensuelle-tgv-aqst/exports/csv?delimiter=%3B&lang=fr&timezone=Europe%2FParis" \
		-o $(DATA_RAW)
	@echo "Downloaded: $(DATA_RAW)"

inject: $(DATA_RAW)
	$(PYTHON) tools/noise_injection.py --input $(DATA_RAW) --output $(DATA_NOISY) --seed 42
	@echo "Noisy dataset saved: $(DATA_NOISY)"

run: $(DATA_NOISY)
	$(PYTHON) main.py --input $(DATA_NOISY) --output $(DATA_CLEANED)

eval: $(DATA_CLEANED)
	$(PYTHON) tools/evaluator.py --noisy $(DATA_NOISY) --cleaned $(DATA_CLEANED)

notebook:
	jupyter notebook notebooks/

clean:
	rm -f $(DATA_NOISY) $(DATA_CLEANED)
	rm -f results/*.json results/*.png
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	@echo "Cleaned generated files."
