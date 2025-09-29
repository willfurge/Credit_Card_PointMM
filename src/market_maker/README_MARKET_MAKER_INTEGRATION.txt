
Market Maker — Config + Logger Integration
------------------------------------------

Run from config with env overrides:
    python -m src.market_maker --config config.json --env-prefix APP_

Examples:
    APP_CONTROLLER__TARGET_RESERVE_RATIO=0.65 \
    APP_SIMULATION__DAYS=180 \
    APP_MODEL__BASE_REDEMPTION_RATE=0.0025 \
    python -m src.market_maker --config config.json

Artifacts:
- data/sim_output.json   (full object dump)
- data/sim_history.csv   (flat table per day)

Logging:
- logs/app.log (level from config.logging.level)

CI:
- .github/workflows/ci.yml runs mypy + pytest on push/PR
