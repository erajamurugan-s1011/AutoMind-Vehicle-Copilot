"""
AutoMind — decision log viewer.

Run:
    python -m src.agent.view_logs                 # last 20 events, all sessions
    python -m src.agent.view_logs --session <id>   # only one session's trace
    python -m src.agent.view_logs --limit 50       # more events
"""

import argparse
from src.agent.logger import read_recent_events


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--session", type=str, default=None, help="Filter to one session_id")
    parser.add_argument("--limit", type=int, default=20, help="Number of recent events to show")
    args = parser.parse_args()

    events = read_recent_events(limit=args.limit, session_id=args.session)

    if not events:
        print("No decision logs found yet — run a chat request first.")
        return

    print(f"📋 Showing last {len(events)} agent decisions"
          f"{f' for session {args.session}' if args.session else ''}:\n")

    for e in events:
        print(f"[{e['timestamp']}] session={e['session_id'][:8]} "
              f"node={e['node']:<20} latency={e['latency_sec']}s")
        print(f"    in:  {e['input']}")
        print(f"    out: {e['output']}")
        print()


if __name__ == "__main__":
    main()