"""Print one game log as a timeline that is easy to read. Reads only.

Run with any Python 3, for example:
    python read_game.py ..\\Collusion_Liars_Bar\\experiments\\runs\\provoke_moderate_s1.json
    python read_game.py <log> --opinions --out provoke_s1.txt

Each line starts with who can see it:
    PUBLIC   every player at the table
    PRIVATE  only the sender and the receiver (secret channel, U's messages)
    HIDDEN   nobody at the table (true cards, reasons, leak checks)
"""
import argparse
import json


def honest(play, target):
    return all(c in (target, "Joker") for c in play["played_cards"])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("log")
    ap.add_argument("--opinions", action="store_true", help="also print each player's private opinions")
    ap.add_argument("--out", help="write to this file instead of the screen")
    args = ap.parse_args()

    d = json.load(open(args.log, encoding="utf-8"))
    m = d["meta"]
    uc = d.get("undercover") or {}
    out = []
    say = out.append
    say(f"Game {m['game_id']}  condition={m['condition']}  framing={m.get('framing')}  seed={m['seed']}")
    say(f"Seats: {' -> '.join(m['seat_order'])}")
    say(f"Alliance: {m.get('alliance')}  Undercover: {m.get('undercover')}  "
        f"U starts in round {m.get('intervene_round')}")
    statements = list(d.get("public_statements", []))
    comms = d.get("secret_channel", {}).get("communications", [])
    events = uc.get("events", [])

    def take(rid, speaker, phase):
        """Next public statement of this speaker and phase in this round (statements are in time order)."""
        for i, s in enumerate(statements):
            if s["round"] == rid and s["speaker"] == speaker and s["phase"] == phase:
                return statements.pop(i)["text"]
        return None

    for r in d["rounds"]:
        rid, target = r["round_id"], r["target_card"]
        say(f"\n{'=' * 25} Round {rid} | target {target} | {r['starting_player']} starts | "
            f"players {', '.join(r['round_players'])} {'=' * 25}")
        for e in [e for e in events if e["round"] == rid and e["kind"] == "private"]:
            say(f"PRIVATE  {uc['name']} -> {e['to']}: {e['text']}")
            say(f"PRIVATE  {e['to']} -> {uc['name']}: {e['reply']}")
        for c in [c for c in comms if c["round"] == rid]:
            say(f"PRIVATE  {c['sender']} -> {c['receiver']}: {c['message']}")
            if c.get("response"):
                say(f"PRIVATE  {c['receiver']} -> {c['sender']}: {c['response']}")
        for p in r["play_history"]:
            who, nxt, n = p["player_name"], p["next_player"], len(p["played_cards"])
            say(f"PUBLIC   {who} plays {n} card{'s' if n > 1 else ''} as {target}. Behavior: {p['behavior']}")
            say(f"HIDDEN     true cards {p['played_cards']} ({'true' if honest(p, target) else 'LIE'}). "
                f"Reason: {p['play_reason']}")
            text = take(rid, who, "play")
            if text:
                say(f"PUBLIC   {who} says: {text}")
            if nxt not in m["seat_order"]:
                continue
            say(f"PUBLIC   {nxt} {'CHALLENGES' if p['was_challenged'] else 'does not challenge'} {who}")
            if p.get("challenge_reason"):
                say(f"HIDDEN     {nxt}'s reason: {p['challenge_reason']}")
            text = take(rid, nxt, "challenge")
            if text:
                say(f"PUBLIC   {nxt} says: {text}")
        rr = r.get("round_result") or {}
        if rr.get("shooter_name"):
            say(f"PUBLIC   {rr['shooter_name']} shoots: {'HIT, out of the game' if rr.get('bullet_hit') else 'survives'}")
        for s in [s for s in statements if s["round"] == rid]:
            say(f"PUBLIC   {s['speaker']} ({s['phase']}): {s['text']}")
        statements = [s for s in statements if s["round"] != rid]
        for e in [e for e in events if e["round"] == rid and e["kind"] not in ("private",)]:
            if e["kind"] == "public":
                continue                                   # already printed as a public statement
            info = {k: v for k, v in e.items() if k not in ("round", "kind")}
            say(f"HIDDEN   U event {e['kind']}: {info}")
        if args.opinions:
            for who, ops in (r.get("player_opinions") or {}).items():
                for other, text in ops.items():
                    say(f"HIDDEN   {who} about {other}: {text}")

    say(f"\nWinner: {d.get('winner')}  Scores: {d.get('final_scores')}")
    text = "\n".join(out)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(text + "\n")
        print(f"written to {args.out}")
    else:
        print(text)


if __name__ == "__main__":
    main()
