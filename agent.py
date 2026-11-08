"""
agent.py - YOUR poker bot. This one file is a complete web server.

Run it:
    python agent.py                  # listens on port 5008
    PORT=5010 python agent.py        # or pick a port

Play a match against a baseline (three windows):
    python agent.py                                 # your bot on port 5008
    python baseline-bots/baseline_random.py         # opponent on port 5009
    python judge_engine.py                          # judge: both halves (uses those two ports)

Check that the tournament can run it:
    python -m pytest -q              # tests in test_agent.py
    ./tools/check_build.sh           # builds and smoke-tests it like the tournament

TODO:
    1. Assign your agent a unique PARTICIPANT and AGENT_NAME below.
    2. Implement the three functions between "YOUR CODE HERE" and "END OF YOUR CODE".

Everything under "SERVER PLUMBING" talks to the judge. You normally never
need to change it, and if you break it the judge can't talk to your bot.
"""

import os
import traceback

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from evaluator import card_rank, card_suit, cards_to_str, omaha_score, warmup  # handy helpers, see evaluator.py
from game import Action, AgentConfig, HandResult, Observation, Street

# Build omaha_score's lookup table now, while the server starts, instead of
# during your first decision (where it would eat into your time bank).
# Do any other slow setup (loading a model, ...) here too: the server must
# answer within 15 seconds of starting.
warmup()

# rename as your team/bot's name
BUILD_TAG = "dh7q_equityedge_k41z"
PARTICIPANT = "DeckHunters"
AGENT_NAME = "EquityEdge"


# ===== YOUR CODE HERE =====
# Strategy "v2": preflop equity table + Monte Carlo equity after the flop,
# pot-odds thresholds, and light opponent modelling (aggression and fold-to-raise rates).

import gc
import sys
import time

# One thread for numpy's math libraries (must be set before numpy is imported).
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_var] = "1"

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import numpy as np  # noqa: E402

from botlib.equity import mc_equity  # noqa: E402  (loads the 18 MB score table at startup)

PRE_RAISE_POT_EQ = 0.60
PRE_RAISE_HALF_EQ = 0.555
PRE_CALL_MARGIN = 0.0
PRE_3BET_EQ = 0.62

SAMPLES = {1: 128, 2: 128, 3: 96}  # tuned version used 200/200/160; cut for the time bank
POST_RAISE_POT_EQ = 0.74
POST_RAISE_HALF_EQ = 0.60
POST_RERAISE_EQ = 0.82
CALL_MARGIN = 0.02
BET_PENALTY = 0.10
BET_FRAC_CAP = 1.5
MAX_RERAISE_STREET = 2

AGG_PRIOR_RAISES = 1.5
AGG_PRIOR_ACTS = 5.0
SCALE_SLOPE = 2.0
SCALE_BASE = 1.5
SCALE_MIN = 0.1

# Time budget per half (the judge's bank is 8 s, charged per request minus a free 1 ms).
# The charge is estimated from this process's CPU time since new_half, which also counts
# the web server's JSON work, minus a part of the free 1 ms per request:
#     est = cpu_used - FREE_CREDIT * requests
# Spending is paced over the half: pace(h) = BANK_TARGET * (h + 1) / hands.
#   est > pace + SOFT_LEAD  -> fewer Monte Carlo samples (REDUCED_SAMPLES)
#   est > pace + HARD_LEAD or est > HARD_CAP or strategy wall time > RAW_HARD -> cheap rule
# The judge's own count runs about 0.4 s above est with everything on one CPU, so
# HARD_CAP 4.6 keeps the judge's count near 5 s even in the worst case.
BANK_TARGET = 4.2
SOFT_LEAD = 0.3
HARD_LEAD = 0.9
HARD_CAP = 4.6
RAW_HARD = 4.0
REDUCED_SAMPLES = 48
FREE_CREDIT = 0.0005
DECISION_DEADLINE = 0.03  # per-decision Monte Carlo wall-clock cap
CHUNK = 64
FOLD_PRIOR_N = 4.0
FOLD_PRIOR_F = 1.2
STEAL_FF = 0.40
BLUFF_FF = 0.45
BLUFF_SLOPE = 1.8
BLUFF_MAX = 0.85
STEAL_GAIN = 0.35
TIGHT_PEN = 0.10
RNG_SEED = 12345
GC_EVERY = 50

_BINOM = [[0] * 6 for _ in range(37)]
for _n in range(37):
    _BINOM[_n][0] = 1
    for _k in range(1, 6):
        if _n:
            _BINOM[_n][_k] = _BINOM[_n - 1][_k - 1] + _BINOM[_n - 1][_k]

try:
    _PRE = np.load(os.path.join(_HERE, "botlib", "data", "preflop_eq.npy"))
except Exception:
    _PRE = None

_RAISES = (Action.RAISE_HALF, Action.RAISE_POT)
_TIME_LOG = os.environ.get("BOT_TIME_LOG") == "1"  # optional stderr timing lines for local checks


def preflop_eq(hole):
    if _PRE is None:
        return 0.5
    h = sorted(hole)
    i = 0
    for k in range(5):
        i += _BINOM[h[k]][k + 1]
    return float(_PRE[i]) / 255.0


def _init_memory(memory):
    memory["rng"] = np.random.default_rng(RNG_SEED)
    memory["opp_acts"] = 0
    memory["opp_raises"] = 0
    memory["my_raises_faced"] = 0
    memory["opp_folds_to_raise"] = 0
    memory["hand"] = -1
    memory["ptr"] = 0
    memory["spent"] = 0.0
    memory["raw"] = 0.0
    memory["cpu0"] = time.process_time()
    memory["calls"] = 1
    memory["errors"] = 0
    memory["cache"] = {}
    memory["pend"] = False
    memory["ended"] = 0


def _warm(memory):
    try:
        mc_equity([0, 1, 2, 3, 4], [5, 6, 7], 8, np.random.default_rng(0))
        mc_equity([0, 1, 2, 3, 4], [5, 6, 7, 8, 9], 8, np.random.default_rng(0))
        preflop_eq([0, 1, 2, 3, 4])
    except Exception:
        pass


def _update_stats(obs, memory):
    if memory.get("hand") != obs.hand_index:
        memory["hand"] = obs.hand_index
        memory["ptr"] = 0
        memory["cache"] = {}
    hist = obs.action_history
    p = memory["ptr"]
    n = len(hist)
    for i in range(p, n):
        e = hist[i]
        if not e.is_me:
            memory["opp_acts"] += 1
            if e.action in _RAISES:
                memory["opp_raises"] += 1
            if i > 0 and hist[i - 1].is_me and hist[i - 1].action in _RAISES:
                memory["my_raises_faced"] += 1
                if e.action == Action.FOLD:
                    memory["opp_folds_to_raise"] += 1
    memory["ptr"] = n


def _fold_freq(memory):
    return (memory["opp_folds_to_raise"] + FOLD_PRIOR_F) / (memory["my_raises_faced"] + FOLD_PRIOR_N)


def _agg(memory):
    return (memory["opp_raises"] + AGG_PRIOR_RAISES) / (memory["opp_acts"] + AGG_PRIOR_ACTS)


def _scale(memory):
    return min(1.0, max(SCALE_MIN, SCALE_BASE - SCALE_SLOPE * _agg(memory)))


def _pick_raise(la, want_pot):
    if want_pot and Action.RAISE_POT in la:
        return Action.RAISE_POT
    if Action.RAISE_HALF in la:
        return Action.RAISE_HALF
    if Action.RAISE_POT in la:
        return Action.RAISE_POT
    return Action.CHECK_CALL


def _bluff(memory, ff):
    p = min(BLUFF_MAX, max(0.0, (ff - BLUFF_FF) * BLUFF_SLOPE))
    return p > 0 and memory["rng"].random() < p


def _decide(obs, memory, eq):
    la = obs.legal_actions
    can_raise = Action.RAISE_HALF in la or Action.RAISE_POT in la
    pre = obs.street == Street.PREFLOP
    to_call = obs.to_call
    nr = obs.raises_this_street
    ff = _fold_freq(memory)
    shift = max(0.0, ff - STEAL_FF) * STEAL_GAIN
    if to_call == 0:
        if not can_raise:
            return Action.CHECK_CALL
        if pre:
            if eq >= PRE_RAISE_POT_EQ:
                return _pick_raise(la, True)
            if eq >= PRE_RAISE_HALF_EQ - shift:
                return _pick_raise(la, False)
        else:
            if eq >= POST_RAISE_POT_EQ:
                return _pick_raise(la, True)
            if eq >= POST_RAISE_HALF_EQ - shift:
                return _pick_raise(la, False)
        if _bluff(memory, ff):
            return _pick_raise(la, False)
        return Action.CHECK_CALL
    need = to_call / float(obs.pot + to_call)
    penalty = 0.0
    if nr > 0 or not pre:
        frac = min(BET_FRAC_CAP, to_call / float(max(1, obs.pot - to_call)))
        penalty = BET_PENALTY * _scale(memory) * frac
        a = _agg(memory)
        if a < 0.3:
            penalty += TIGHT_PEN * (0.3 - a) / 0.3 * frac
    adj = eq - penalty
    if can_raise and nr <= MAX_RERAISE_STREET:
        if pre:
            if adj >= PRE_3BET_EQ:
                return _pick_raise(la, adj >= PRE_3BET_EQ + 0.03)
            if nr == 0 and ((ff >= STEAL_FF and eq >= PRE_RAISE_HALF_EQ - shift) or _bluff(memory, ff)):
                return _pick_raise(la, False)
        else:
            if adj >= POST_RERAISE_EQ:
                return _pick_raise(la, True)
            if nr == 0 and adj >= POST_RAISE_HALF_EQ + 0.08:
                return _pick_raise(la, adj >= POST_RAISE_POT_EQ + 0.04)
    margin = PRE_CALL_MARGIN if pre else CALL_MARGIN
    if adj >= need + margin:
        return Action.CHECK_CALL
    if Action.FOLD in la:
        return Action.FOLD
    return Action.CHECK_CALL


def _equity(obs, memory):
    if obs.street == Street.PREFLOP:
        return preflop_eq(obs.hole_cards)
    key = (tuple(obs.hole_cards), tuple(obs.board))
    c = memory["cache"].get(key)
    if c is not None:
        return c
    mode = _budget_mode(obs, memory)
    if mode == 2:
        return None
    n = SAMPLES.get(int(obs.street), 150)
    if mode == 1:
        n = REDUCED_SAMPLES
    rng = memory["rng"]
    t0 = time.perf_counter()
    tot = 0.0
    done = 0
    while done < n:
        m = min(CHUNK, n - done)
        tot += mc_equity(obs.hole_cards, obs.board, m, rng) * m
        done += m
        if time.perf_counter() - t0 > DECISION_DEADLINE:
            break
    eq = tot / done
    memory["cache"][key] = eq
    return eq


def _budget_mode(obs, memory):
    """0 = full sampling, 1 = reduced sampling, 2 = cheap rule (no Monte Carlo)."""
    est = memory.get("spent", 0.0)
    if est > HARD_CAP or memory.get("raw", 0.0) > RAW_HARD:
        return 2
    pace = BANK_TARGET * (obs.hand_index + 1) / float(max(1, memory.get("n_hands", 1000)))
    if est > pace + HARD_LEAD:
        return 2
    if est > pace + SOFT_LEAD:
        return 1
    return 0


def _cheap(obs):
    la = obs.legal_actions
    if obs.to_call == 0:
        return Action.CHECK_CALL
    need = obs.to_call / float(obs.pot + obs.to_call)
    if need <= 0.25 and Action.CHECK_CALL in la:
        return Action.CHECK_CALL
    return Action.FOLD if Action.FOLD in la else Action.CHECK_CALL


def _safe(obs, a):
    """Final legality check: always hand back one of obs.legal_actions."""
    try:
        if isinstance(a, Action) and a in obs.legal_actions:
            return a
        if Action.CHECK_CALL in obs.legal_actions:
            return Action.CHECK_CALL
        return obs.legal_actions[0]
    except Exception:
        return Action.CHECK_CALL


def new_half(config: AgentConfig, memory: dict) -> None:
    try:
        try:
            _n_hands = int(getattr(config, "num_hands", 1000) or 1000)
        except Exception:
            _n_hands = 1000
        _init_memory(memory)
        # Same warm-up draw from memory["rng"] as the tuned version, so the random stream matches it.
        mc_equity([0, 1, 2, 3, 4], [5, 6, 7], 8, memory["rng"])
        _warm(memory)
        memory["n_hands"] = _n_hands
        gc.collect()
        gc.freeze()
    except Exception:
        pass


def choose_action(obs: Observation, memory: dict) -> Action:
    t0 = time.perf_counter()
    a = Action.CHECK_CALL
    try:
        if "rng" not in memory:
            _init_memory(memory)
        memory["pend"] = False
        _update_stats(obs, memory)
        eq = _equity(obs, memory)
        if eq is None:
            a = _cheap(obs)
        else:
            a = _decide(obs, memory, eq)
        a = _safe(obs, a)
        memory["pend"] = a in _RAISES
    except Exception:
        a = _safe(obs, Action.CHECK_CALL)
        try:
            memory["errors"] = memory.get("errors", 0) + 1
        except Exception:
            pass
    try:
        _dt = time.perf_counter() - t0
        memory["raw"] = memory.get("raw", 0.0) + _dt
        if _dt > memory.get("maxdt", 0.0):
            memory["maxdt"] = _dt
        memory["calls"] = memory.get("calls", 1) + 1
        cpu = time.process_time() - memory.get("cpu0", 0.0)
        memory["spent"] = max(0.0, cpu - FREE_CREDIT * memory["calls"])
    except Exception:
        pass
    return a


def on_hand_end(result: HandResult, memory: dict) -> None:
    try:
        if memory.get("pend"):
            memory["pend"] = False
            memory["my_raises_faced"] = memory.get("my_raises_faced", 0) + 1
            if not result.went_to_showdown and result.my_net > 0:
                memory["opp_folds_to_raise"] = memory.get("opp_folds_to_raise", 0) + 1
        memory["ended"] = memory.get("ended", 0) + 1
        if _TIME_LOG and memory["ended"] % 250 == 0:
            print(f"[time] hands={memory['ended']} est_charge={memory.get('spent', 0.0):.3f}s "
                  f"strategy_wall={memory.get('raw', 0.0):.3f}s max_decision={1000 * memory.get('maxdt', 0.0):.1f}ms "
                  f"calls={memory.get('calls', 0)}",
                  file=sys.stderr, flush=True)
        if memory["ended"] % GC_EVERY == 0:
            gc.collect(1)
    except Exception:
        pass


# Startup: tables are loaded above; warm numpy, then freeze the startup heap and
# switch off automatic garbage collection (collected every GC_EVERY hands instead).
_warm({})
gc.collect()
gc.freeze()
gc.disable()

# ===== END OF YOUR CODE =====


# ===== SERVER PLUMBING (you normally don't need to edit below) =====
# The judge calls these routes:
#   GET  /           -> who you are (also used as the health check)
#   POST /new-half   -> start a fresh half: memory is wiped
#   POST /act        -> the judge sends any finished hands plus an Observation;
#                       on_hand_end() runs for each hand, then you answer an action
#   POST /hand-end   -> the judge sends the last finished hands of a half
#
# Why "async def"? The routes run on one thread and call your functions
# directly, so your functions never run twice at once. Your own functions
# stay normal "def" (don't make them async).
#
# Keep your per-half state in `memory`. In the tournament every half runs in
# a fresh server process, so everything starts over. When you test locally,
# though, one process can play both halves and only `memory` is wiped, NOT
# other global variables. Using `memory` makes your bot behave the same way
# in both places.

app = FastAPI()
MEMORY: dict = {}                                   # your memory for this half
STATUS = {"started": False}                         # has /new-half been called?
TOKEN = os.environ.get("AGENT_TOKEN") or None       # set by the tournament


async def _read_request(request: Request, path: str) -> tuple[dict, JSONResponse | None]:
    """Check the token (and that the half has started) BEFORE reading the body.

    Checking first means a request without the right token costs your bot
    almost nothing, so nobody else can slow it down by sending junk.
    Returns (body, None) when the request is fine, or ({}, error response).
    """
    if TOKEN and request.headers.get("x-agent-token") != TOKEN:
        return {}, JSONResponse({"error": "bad or missing X-Agent-Token"}, status_code=401)
    if not STATUS["started"] and path != "/new-half":
        return {}, JSONResponse({"error": "call /new-half first"}, status_code=409)
    try:
        body = await request.json()
    except ValueError:
        body = None
    if not isinstance(body, dict):
        return {}, JSONResponse({"error": "the request body must be a JSON object"}, status_code=400)
    return body, None


def _error_info(exc: Exception) -> dict:
    """Describe an exception from your code (call this inside `except`)."""
    return {"error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()}


def _crash(exc: Exception, hand_end_errors: list | None = None) -> JSONResponse:
    """Report an exception from your code to the judge (with traceback)."""
    return JSONResponse({**_error_info(exc), "hand_end_errors": hand_end_errors or []},
                        status_code=500)


def _deliver_results(raw_results) -> list:
    """Call on_hand_end() for each finished hand the judge sent.
    Returns a list of error reports (one per hand that crashed)."""
    errors = []
    for raw in raw_results or []:
        try:
            on_hand_end(HandResult.from_dict(raw), MEMORY)
        except Exception as exc:
            errors.append(_error_info(exc))
    return errors


@app.get("/")
async def info():
    """Health check: tells the judge who this bot is."""
    return {"participant": PARTICIPANT, "agent_name": AGENT_NAME}


@app.post("/new-half")
async def route_new_half(request: Request):
    """Wipe memory and call your new_half(). The body is {"config": {...}}."""
    body, refused = await _read_request(request, "/new-half")
    if refused:
        return refused
    MEMORY.clear()
    STATUS["started"] = True
    try:
        new_half(AgentConfig.from_dict(body.get("config", {})), MEMORY)
    except Exception as exc:
        return _crash(exc)
    return {"ok": True}


@app.post("/act")
async def route_act(request: Request):
    """Deliver finished hands, then turn the JSON into an Observation, call
    choose_action(), and send the answer back."""
    body, refused = await _read_request(request, "/act")
    if refused:
        return refused
    hand_end_errors = _deliver_results(body.get("hand_results"))
    try:
        obs = Observation.from_dict(body["observation"])
        action = choose_action(obs, MEMORY)
    except Exception as exc:
        return _crash(exc, hand_end_errors)
    # Send the action's name; anything strange is passed on for the judge to reject.
    if isinstance(action, Action):
        answer = action.name
    elif action is None or isinstance(action, (str, int, float, bool)):
        answer = action
    else:
        answer = repr(action)
    return {"action": answer, "hand_end_errors": hand_end_errors}


@app.post("/hand-end")
async def route_hand_end(request: Request):
    """Call on_hand_end() for the last finished hands of a half."""
    body, refused = await _read_request(request, "/hand-end")
    if refused:
        return refused
    return {"ok": True, "hand_end_errors": _deliver_results(body.get("results"))}


if __name__ == "__main__":
    # The tournament sets PORT as an environment variable, so each bot is compliant to our evaluation platform
    # Locally it defaults to 5008 for your own testing convenience
    port = int(os.environ.get("PORT", "5008"))
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info", access_log=False)
