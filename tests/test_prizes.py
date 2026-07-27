obs = {'current': {'yourIndex': 0, 'players': [{'prize': [1,2,3,4,5,6]}, {'prize': [1,2,3,4,5,6]}]}}
current = obs.get("current", {})
players = current.get("players", [])
p1_idx = current.get("yourIndex", 0)
p2_idx = 1 - p1_idx
p1_cur = len(players[p1_idx].get("prize", []))
p2_cur = len(players[p2_idx].get("prize", []))
print(f"p1_cur: {p1_cur}, p2_cur: {p2_cur}")
