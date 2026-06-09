.PHONY: dev prod monitoring cve-scan down lint

dev:
	CADDYFILE=./Caddyfile.dev \
	COMPOSE_PROFILES=dev \
	docker compose up --build

prod:
	COMPOSE_PROFILES=prod \
	docker compose up -d

monitoring:
	COMPOSE_PROFILES=prod,monitoring \
	docker compose up -d

cve-scan:
	COMPOSE_PROFILES=scan \
	docker compose run --rm cve-scan

down:
	docker compose --profile dev --profile prod --profile monitoring down --remove-orphans -v

lint:
	cd frontend && ng lint