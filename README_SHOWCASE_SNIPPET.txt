
## Demo

### Quick Run
```bash
python -m src.market_maker --config config.json
python scripts/plots.py  # saves PNGs under assets/
```

### Streamlit (Interactive)
```bash
pip install streamlit pandas matplotlib
streamlit run streamlit_app.py
```

This launches a dashboard where you can:
- Adjust controller targets and gains
- Change spend and redemption cost assumptions
- Run a fresh simulation and visualize reserve ratio, quotes, and balance sheet
- Download the CSV/JSON outputs
