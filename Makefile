deploy:
	@echo "Running CVE scan..."
	docker compose --profile scan run --rm cve-scan
	@echo "Scan passed. Bringing up stack..."
	docker compose up -d

.PHONY: deploy