# TAMU Datathon 2026 Heads Up Poker Bot: EquityEdge (DeckHunters)

Heads-up Short Deck 5-card Pot-Limit Omaha (PLO5) bot for the TAMU Datathon 2026 Heads Up challenge, built on the official heads-up-bundle. Keywords: tamu datathon 2026, heads up, heads-up-bundle, short deck omaha bot, plo bot short deck, datathon poker.

Equity + pot odds bot for heads-up short deck PLO5. Preflop lookup table, Monte Carlo equity postflop, pot-odds calls. v3 coming (better sizing).

Tournament bot for the TAMU Datathon 2026 poker challenge, built on the official participant bundle.

- `agent.py`: the bot (strategy between the `YOUR CODE HERE` markers, server plumbing unchanged).
- `botlib/`: Omaha score table (`data/omaha_table.npy`, 2 hole cards x 3 board cards, checked against `evaluator.py` at startup), Monte Carlo equity, and the preflop equity table (`data/preflop_eq.npy`).
- Everything else comes from the official bundle (`game.py`, `evaluator.py`, `judge_engine.py`, `baseline-bots/`, `test_agent.py`, `docker/`, `tools/`).

Run locally:

```
pip install -r requirements-dev.txt
python agent.py                            # port 5008
python baseline-bots/baseline_random.py    # port 5009
python judge_engine.py                     # 1000 deals, both halves
python -m pytest -q test_agent.py
```
