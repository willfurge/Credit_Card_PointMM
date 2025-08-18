# Credit Card Point Market Maker

This project models credit card reward points as a market and implements reserving and market-making logic to manage issuance, redemption, and reserve adequacy. The goal is to simulate how a bank would set reserves and policy levers to keep the point economy balanced and solvent.

## Features

- Market simulation engine for issuance and redemption flows
- Reserving methods to estimate required reserves under uncertainty
- Streamlit dashboard for interactive monitoring
- Test suite for core components

## Project Structure

.
├── src/                   # Core source code
│   ├── data_pipeline/     # ETL and data preparation
│   ├── models/            # Reserving and simulation models
│   └── utils/             # Shared utilities
├── tests/                 # Unit tests
├── docs/                  # Project documentation
├── data/                  # Raw/processed data (gitignored)
├── logs/                  # Application logs (gitignored)
├── notebooks/             # Jupyter notebooks (experiments, EDA)
├── run_dashboard.py       # Streamlit app entrypoint
├── app.py                 # Script entrypoint if applicable
├── requirements.txt       # Python dependencies
├── pyproject.toml         # Build and tool configuration
└── README.md

Note: The folders under data/ and logs/ are ignored by git and are created at runtime as needed.

## Installation

1) Clone the repository:

git clone https://github.com/willfurge/Credit_Card_PointMM.git
cd Credit_Card_PointMM

2) Create and activate a virtual environment:

python -m venv .venv
# macOS/Linux:
source .venv/bin/activate
# Windows (PowerShell):
.venv\Scripts\Activate.ps1

3) Install dependencies:

pip install -r requirements.txt

## Usage

Run the Streamlit dashboard:

streamlit run run_dashboard.py

Run the main script (if used):

python app.py

## Tests

Run tests with pytest:

pytest

## Development Notes

- Never commit secrets. Store environment variables in a local .env file that is not checked into git. Provide a .env.example if configuration keys are required.
- Keep notebooks light and consider exporting important logic to src/ so it can be tested.
- Prefer small, focused pull requests with clear commit messages.
- Pre-commit configuration is included; install hooks with:
  pre-commit install
  pre-commit run --all-files

## Roadmap

- Additional reserving methods (stochastic and scenario-based)
- Historical backtesting on redemption behavior
- Enhanced dashboard charts and drill-downs
- CI workflow to run tests and style checks on push

## License

Apache 2 License
