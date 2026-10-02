#!/usr/bin/env python3
"""Make the walking graph follow the paths that exist in the aerial, then attach every room to it.

  1. Edges that run far from any detected path (schematic links across parking lots, roofs, the lake)
     are rerouted as least-cost paths over the aerial path raster (cheap on path/asphalt pixels,
     expensive on desert, very expensive inside buildings).
  2. Every room and landmark is re-attached to the nearest point of the nearest edge (edge split).
  3. New nodes are converted back to PDF coordinates with a Newton inverse of the forward grid, so the
     app (which stores nodes in PDF space) renders them exactly where they were computed.

  python3 tools/finalize_graph.py --data data/appdata.final.json
"""
import sys, json, pathlib, argparse, math, time
import numpy as np
from scipy.spatial import cKDTree
from skimage.graph import route_through_array
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from aerial import root, m2ll, ll2m, LAT0, LON0, K_LAT, K_LON

ap = argparse.ArgumentParser(); ap.add_argument("--data", default="data/appdata.final.json"); ap.add_argument("--offpath", type=float, default=6.0); ap.add_argument("--no-reroute", action="store_true", help="keep every drawn stroke; only drop dead-end spurs that end inside water, courts or buildings"); ap.add_argument("--obstacles-only", action="store_true", help="reroute only strokes that run >12 m through water, courts or buildings; keep all other drawn strokes"); args = ap.parse_args()
t0 = time.time()
D = json.load(open(root / args.data)); G = D["graph"]; rooms = D["rooms"]; landmarks = D["landmarks"]
A = np.load(root / "data/aerial_paths.npz"); dpath, dbld = A["dpath"], A["dbld"]; tx0, ty0, ts, z = [int(v) for v in A["geo"]]; res = float(A["res"]); n_t = 2 ** z; H, W = dpath.shape

def grid_lookup(g, X):
    X = np.atleast_2d(X); v = np.asarray(g["v"]).reshape(g["ny"], g["nx"], 2); fx = (X[:, 0] - g["x0"]) / g["step"]; fy = (X[:, 1] - g["y0"]) / g["step"]
    i = np.clip(np.floor(fx).astype(int), 0, g["nx"] - 2); j = np.clip(np.floor(fy).astype(int), 0, g["ny"] - 2); tx = (fx - i)[:, None]; ty = (fy - j)[:, None]
    return (1 - tx) * (1 - ty) * v[j, i] + tx * (1 - ty) * v[j, i + 1] + (1 - tx) * ty * v[j + 1, i] + tx * ty * v[j + 1, i + 1]
fwd = lambda X: grid_lookup(D["fwd"], X)
def inv_exact(M, iters=8):
    """metres -> PDF points: start from the inverse grid, polish with Newton on the bilinear forward grid."""
    P = grid_lookup(D["inv"], M).copy(); h = 0.5
    for _ in range(iters):
        F = fwd(P) - M; Jx = (fwd(P + [h, 0]) - fwd(P - [h, 0])) / (2 * h); Jy = (fwd(P + [0, h]) - fwd(P - [0, h])) / (2 * h)
        det = Jx[:, 0] * Jy[:, 1] - Jx[:, 1] * Jy[:, 0]; det = np.where(np.abs(det) < 1e-9, 1e-9, det)
        dx = (F[:, 0] * Jy[:, 1] - F[:, 1] * Jy[:, 0]) / det; dy = (Jx[:, 0] * F[:, 1] - Jx[:, 1] * F[:, 0]) / det
        P -= np.c_[dx, dy]
    return P
def m2px(M):
    M = np.atleast_2d(M); lat = LAT0 + M[:, 1] / K_LAT; lon = LON0 + M[:, 0] / K_LON; x = (lon + 180) / 360 * n_t; lr = np.radians(lat)
    y = (1 - np.log(np.tan(lr) + 1 / np.cos(lr)) / math.pi) / 2 * n_t; return np.c_[(x - tx0) * ts, (y - ty0) * ts]
def px2m(P):
    P = np.atleast_2d(P); x = P[:, 0] / ts + tx0; y = P[:, 1] / ts + ty0; lon = x / n_t * 360 - 180; lat = np.degrees(np.arctan(np.sinh(math.pi * (1 - 2 * y / n_t))))
    return ll2m(lat, lon)

nodesP = np.array(G["nodes"], float); nodesM = fwd(nodesP); edges = [tuple(e) for e in G["edges"]]
print(f"{len(nodesM)} nodes, {len(edges)} edges; nodes->pdf roundtrip check: {np.abs(inv_exact(nodesM) - nodesP).max():.3f} pt")

# ---------- obstacles: buildings (distance field) plus OSM water and sports pitches ----------
import cv2
osm = json.load(open(root / "data/osm.json"))["elements"]; obst = (dbld == 0).astype(np.uint8)
for e in osm:
    t = e.get("tags", {}); g = e.get("geometry")
    if not g or len(g) < 4: continue
    if t.get("natural") == "water" or t.get("leisure") in ("swimming_pool", "pitch") or t.get("water"):
        cv2.fillPoly(obst, [m2px(ll2m([q["lat"] for q in g], [q["lon"] for q in g])).astype(np.int32)], 1)
MANUAL_OBSTACLES = json.load(open(root / "data/obstacles.json")) if (root / "data/obstacles.json").exists() else []
for poly in MANUAL_OBSTACLES: cv2.fillPoly(obst, [m2px(np.array(poly, float)).astype(np.int32)], 1)
print(f"obstacle mask: {obst.mean():.2%} of area ({len(MANUAL_OBSTACLES)} manual polygons)")
# ---------- cost raster ----------
cost = 1.0 + np.minimum((dpath / 2.5) ** 2, 400.0); cost[obst > 0] += 150.0   # inside a footprint: strongly discouraged (but not impossible: lodge interiors)
def sample_dpath(a, b):
    P = np.c_[np.linspace(a[0], b[0], 12), np.linspace(a[1], b[1], 12)]; q = m2px(P).round().astype(int); q[:, 0] = np.clip(q[:, 0], 0, W - 1); q[:, 1] = np.clip(q[:, 1], 0, H - 1); return dpath[q[:, 1], q[:, 0]]
def near_building(m): q = m2px(m).round().astype(int)[0]; return dbld[np.clip(q[1], 0, H - 1), np.clip(q[0], 0, W - 1)] < 12
def inside_frac(a, b):
    P = np.c_[np.linspace(a[0], b[0], 12), np.linspace(a[1], b[1], 12)]; q = m2px(P).round().astype(int); q[:, 0] = np.clip(q[:, 0], 0, W - 1); q[:, 1] = np.clip(q[:, 1], 0, H - 1); return (obst[q[:, 1], q[:, 0]] > 0).mean()
def reroute(a, b):
    pa, pb = m2px(a)[0], m2px(b)[0]; margin = int(200 / res)
    x0 = int(max(0, min(pa[0], pb[0]) - margin)); x1 = int(min(W, max(pa[0], pb[0]) + margin)); y0 = int(max(0, min(pa[1], pb[1]) - margin)); y1 = int(min(H, max(pa[1], pb[1]) + margin))
    sub = cost[y0:y1, x0:x1]; s = (int(pa[1]) - y0, int(pa[0]) - x0); e = (int(pb[1]) - y0, int(pb[0]) - x0)
    s = (min(max(s[0], 0), sub.shape[0] - 1), min(max(s[1], 0), sub.shape[1] - 1)); e = (min(max(e[0], 0), sub.shape[0] - 1), min(max(e[1], 0), sub.shape[1] - 1))
    path, c = route_through_array(sub, s, e, fully_connected=True, geometric=True)
    P = np.array(path, float)[:, ::-1] + [x0 + 0.5, y0 + 0.5]
    return px2m(P), c
def simplify(P, tol):
    if len(P) < 3: return P
    a, b = P[0], P[-1]; ab = b - a; L = np.linalg.norm(ab)
    d = np.abs(np.cross(ab, P - a)) / L if L > 1e-9 else np.linalg.norm(P - a, axis=1); i = int(d.argmax())
    if d[i] > tol: return np.vstack([simplify(P[:i + 1], tol)[:-1], simplify(P[i:], tol)])
    return np.vstack([a, b])

# ---------- chains: collapse runs of degree-2 nodes so a schematic stroke is judged as a whole ----------
from collections import defaultdict
adj = defaultdict(set)
for a, b in edges: adj[a].add(b); adj[b].add(a)
keep_node = {v for v in G.get("roomNode", {}).values()} | {v for v in G.get("placeNode", {}).values()}
is_junction = lambda v: len(adj[v]) != 2 or v in keep_node
seen = set(); chains = []
for v in range(len(nodesM)):
    if not is_junction(v): continue
    for w in adj[v]:
        if (v, w) in seen: continue
        chain = [v, w]; seen.add((v, w)); seen.add((w, v))
        while not is_junction(chain[-1]):
            nxt = [u for u in adj[chain[-1]] if u != chain[-2]]
            if not nxt: break
            u = nxt[0]; seen.add((chain[-1], u)); seen.add((u, chain[-1])); chain.append(u)
            if u == v: break
        chains.append(chain)
# isolated cycles of degree-2 nodes
for v in range(len(nodesM)):
    for w in adj[v]:
        if (v, w) not in seen:
            chain = [v, w]; seen.add((v, w)); seen.add((w, v))
            while True:
                nxt = [u for u in adj[chain[-1]] if (chain[-1], u) not in seen]
                if not nxt: break
                u = nxt[0]; seen.add((chain[-1], u)); seen.add((u, chain[-1])); chain.append(u)
            chains.append(chain)
def split_chain(ch, piece=40.0):
    out, cur, acc = [], [ch[0]], 0.0
    for u, v in zip(ch[:-1], ch[1:]):
        acc += np.linalg.norm(nodesM[v] - nodesM[u]); cur.append(v)
        if acc >= piece and v != ch[-1]: out.append(cur); cur, acc = [v], 0.0
    if len(cur) > 1: out.append(cur)
    return out
chains = [piece for ch in chains for piece in split_chain(ch)]
print(f"{len(chains)} chains (after splitting into ~40 m pieces)")
def obst_run(ch):
    """longest contiguous run (metres) of the chain inside an obstacle, sampled every metre."""
    P = nodesM[ch]; pts = []
    for a, b in zip(P[:-1], P[1:]):
        n = max(2, int(np.linalg.norm(b - a))); pts.append(np.c_[np.linspace(a[0], b[0], n), np.linspace(a[1], b[1], n)])
    pts = np.vstack(pts); q = m2px(pts).round().astype(int); q[:, 0] = np.clip(q[:, 0], 0, W - 1); q[:, 1] = np.clip(q[:, 1], 0, H - 1)
    inside = obst[q[:, 1], q[:, 0]] > 0; best = run = 0
    for v in inside: run = run + 1 if v else 0; best = max(best, run)
    return best
def in_obst(m): q = m2px(m).round().astype(int)[0]; return obst[np.clip(q[1], 0, H - 1), np.clip(q[0], 0, W - 1)] > 0
def chain_stats(ch):
    P = nodesM[ch]; segs = list(zip(P[:-1], P[1:])); L = sum(np.linalg.norm(b - a) for a, b in segs)
    dp = np.concatenate([sample_dpath(a, b) for a, b in segs]); ins = np.mean([inside_frac(a, b) for a, b in segs]); return L, dp, ins
new_nodes = list(map(tuple, nodesM)); new_edges = []; rerouted = 0; dropped = 0; kept = 0; dropped_chains = []
edge_src = []            # per new edge: None for a drawn stroke, else the chain (list of original nodes) it replaced
DEBUG_BOX = [float(v) for v in __import__("os").environ.get("DEBUG_BOX", "").split(",")] if __import__("os").environ.get("DEBUG_BOX") else None
def dbg(ch, what, extra=""):
    if not DEBUG_BOX: return
    P = nodesM[ch]; x0, x1, y0, y1 = DEBUG_BOX
    if ((P[:, 0] > x0) & (P[:, 0] < x1) & (P[:, 1] > y0) & (P[:, 1] < y1)).any():
        L, dp, ins = chain_stats(ch); print(f"    [{what}] chain {ch[0]}..{ch[-1]} n={len(ch)} L={L:.0f} offpath={np.mean(dp > args.offpath):.0%} ins={ins:.2f} orun={obst_run(ch):.0f} deg={len(adj[ch[0]])},{len(adj[ch[-1]])} keep={ch[0] in keep_node or ch[-1] in keep_node} from {nodesM[ch[0]].round(0)} to {nodesM[ch[-1]].round(0)} {extra}")
for ch in chains:
    L, dp, ins = chain_stats(ch); a, b = ch[0], ch[-1]; orun = obst_run(ch)
    # a dead-end stroke that ends inside water / a court / a building nobody is routed to is fiction
    if (len(adj[a]) <= 1 and in_obst(nodesM[a]) and a not in keep_node) or (len(adj[b]) <= 1 and in_obst(nodesM[b]) and b not in keep_node):
        dropped += 1; dbg(ch, "drop-deadend"); continue
    if args.obstacles_only: schematic = L > 12 and orun > 12
    else: schematic = (not args.no_reroute) and L > 12 and ((dp > args.offpath).mean() > 0.5 or ins > 0.15 or orun > 8)
    if not schematic:
        for u, v in zip(ch[:-1], ch[1:]): new_edges.append((u, v)); edge_src.append(None)
        kept += 1; dbg(ch, "keep"); continue
    if a == b: dropped += 1; continue
    P, c = reroute(nodesM[a], nodesM[b]); P = simplify(P, 1.0); Lr = np.linalg.norm(np.diff(P, axis=0), axis=1).sum()
    through = ins > 0.15 or orun > 8
    if (c / max(Lr, 1) > 200 and not through) or Lr > 3.0 * L + 30:      # no real path connects these ends (a line through an obstacle is rerouted regardless of cost)
        if through and (len(adj[a]) <= 1 or len(adj[b]) <= 1): dropped += 1; dropped_chains.append((ch, L)); dbg(ch, "drop-spur"); continue   # spur into a building: drop
        dbg(ch, "keep-noroute", f"c/Lr={c / max(Lr, 1):.0f} Lr={Lr:.0f}")
        for u, v in zip(ch[:-1], ch[1:]): new_edges.append((u, v)); edge_src.append(None)                                # otherwise keep the drawing's line
        kept += 1; continue
    # still through an obstacle after rerouting (e.g. the far side is only reachable across the lake): fiction, drop
    q = m2px(P).round().astype(int); q[:, 0] = np.clip(q[:, 0], 0, W - 1); q[:, 1] = np.clip(q[:, 1], 0, H - 1)
    ins_pts = obst[q[:, 1], q[:, 0]] > 0; run = best = 0
    for k in range(1, len(P)):
        run = run + np.linalg.norm(P[k] - P[k - 1]) if ins_pts[k] else 0; best = max(best, run)
    if best > 8: dropped += 1; dropped_chains.append((ch, L)); dbg(ch, "drop-still-obst", f"Lr={Lr:.0f} best={best:.0f}"); continue
    dbg(ch, "reroute", f"Lr={Lr:.0f}")
    idx = [a]
    for q in P[1:-1]: new_nodes.append(tuple(q)); idx.append(len(new_nodes) - 1)
    idx.append(b)
    for u, v in zip(idx[:-1], idx[1:]): new_edges.append((u, v)); edge_src.append(ch)
    rerouted += 1
print(f"chains kept {kept}, rerouted {rerouted}, dropped {dropped}; {len(new_nodes)} nodes ({time.time() - t0:.0f}s)")
# a dropped stroke whose ends are now far apart on the graph was a real connection the path detector missed (sand on sand): restore it
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra
def graph_dist(nodes, eds, a, b):
    nodes = np.asarray(nodes); eds = np.asarray(eds); w = np.linalg.norm(nodes[eds[:, 0]] - nodes[eds[:, 1]], axis=1)
    A = coo_matrix((w, (eds[:, 0], eds[:, 1])), shape=(len(nodes), len(nodes))); return dijkstra(A, directed=False, indices=a)[b]
restored = 0
deg = np.bincount(np.asarray(new_edges).flatten(), minlength=len(new_nodes))
for ch, L in sorted(dropped_chains, key=lambda x: -x[1]):
    if deg[ch[0]] == 0 or deg[ch[-1]] == 0: continue                # a dead-end spur: nothing to reconnect
    d = graph_dist(new_nodes, new_edges, ch[0], ch[-1])
    if not np.isfinite(d) or d > 2.5 * L + 40:
        for u, v in zip(ch[:-1], ch[1:]): new_edges.append((u, v)); edge_src.append(None)
        restored += 1
print(f"restored {restored} dropped strokes that were real connections ({time.time() - t0:.0f}s)")

# ---------- planarity: rerouted pieces must not cross other paths ----------
def seg_cross(p1, p2, p3, p4):
    d = lambda a, b, c: (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
    d1, d2, d3, d4 = d(p3, p4, p1), d(p3, p4, p2), d(p1, p2, p3), d(p1, p2, p4)
    return (d1 * d2 < 0) and (d3 * d4 < 0)
for _ in range(3):
    NMa = np.array(new_nodes); Ea = np.array(new_edges); mid = (NMa[Ea[:, 0]] + NMa[Ea[:, 1]]) / 2; half = np.linalg.norm(NMa[Ea[:, 0]] - NMa[Ea[:, 1]], axis=1) / 2
    tree = cKDTree(mid); bad = set()
    for i in range(len(Ea)):
        for j in tree.query_ball_point(mid[i], half[i] + 30):
            if j <= i or len({*Ea[i]} & {*Ea[j]}): continue
            if seg_cross(NMa[Ea[i, 0]], NMa[Ea[i, 1]], NMa[Ea[j, 0]], NMa[Ea[j, 1]]):
                for k in (i, j):
                    if edge_src[k] is not None: bad.add(id(edge_src[k]))
    if not bad: break
    revert = {id(c): c for c in edge_src if c is not None and id(c) in bad}
    keep_idx = [k for k in range(len(new_edges)) if edge_src[k] is None or id(edge_src[k]) not in bad]
    new_edges = [new_edges[k] for k in keep_idx]; edge_src = [edge_src[k] for k in keep_idx]
    for ch in revert.values():
        for u, v in zip(ch[:-1], ch[1:]): new_edges.append((u, v)); edge_src.append(None)
    print(f"  {len(revert)} rerouted pieces crossed another path: reverted to the drawn stroke")
# ---------- attach rooms and landmarks: nearest point on the nearest edge, split there ----------
NM = np.array(new_nodes); E = np.array(new_edges)
def nearest_on_edges(p):
    Aa, Bb = NM[E[:, 0]], NM[E[:, 1]]; ab = Bb - Aa; t = np.clip(((p - Aa) * ab).sum(1) / np.maximum((ab * ab).sum(1), 1e-9), 0, 1)
    Q = Aa + t[:, None] * ab; d = np.linalg.norm(Q - p, axis=1); i = int(d.argmin()); return i, t[i], Q[i], d[i]
attach = {}
def attach_point(p):
    global NM, E
    i, t, Q, d = nearest_on_edges(p)
    if t < 0.02: return int(E[i, 0])
    if t > 0.98: return int(E[i, 1])
    NM = np.vstack([NM, Q]); k = len(NM) - 1; a, b = E[i]; E = np.vstack([np.delete(E, i, 0), [[a, k], [k, b]]]); return k
dists = []
orig_room = dict(G["roomNode"]); orig_place = dict(G["placeNode"]); deg0 = np.bincount(E.flatten(), minlength=len(NM))
for k, v in rooms.items():
    p = ll2m(v[0], v[1])[0]; i, t, Q, d = nearest_on_edges(p); dists.append(d)
    o = orig_room.get(k); attach[k] = int(o) if o is not None and deg0[o] > 0 else attach_point(p)
print(f"rooms keeping the drawing's connection: {sum(1 for k in rooms if orig_room.get(k) is not None and deg0[orig_room[k]] > 0)} of {len(rooms)}")
print(f"rooms attached: distance to path median {np.median(dists):.1f} m, p90 {np.percentile(dists, 90):.1f}, max {max(dists):.1f} ({time.time() - t0:.0f}s)")
EXACT_LM = {"The Spa & Spa Cafe": (-626.0, 336.0), "Lodge (Main Lobby)": (-330.0, 60.0), "Lodge Pool": (-390.6, 40.9), "Spa Pool": (-587.5, 305.3), "Golf Clubhouse": (259.2, 235.9),
            "Club Lake": (303.0, 160.0), "Duck Pond": (-296.0, 13.0), "Tennis Center": (165.0, 180.0), "Main Gate": (-705.7, 415.5), "Discovery Lounge": (-340.0, 40.0), "Palo Verde Restaurant": (-335.0, 18.0)}
for k, m in EXACT_LM.items():
    if k in landmarks: ll = m2ll(np.array([m], float))[0]; landmarks[k][0] = round(float(ll[0]), 6); landmarks[k][1] = round(float(ll[1]), 6)
pattach = {k: (int(orig_place[k]) if k in orig_place and deg0[orig_place[k]] > 0 and k not in EXACT_LM else attach_point(ll2m(v[0], v[1])[0])) for k, v in landmarks.items()}
# connectivity: every component joins the main one through its closest node pair (an interior corridor or a short unmapped link)
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
for _ in range(50):
    cc, lab = connected_components(coo_matrix((np.ones(len(E)), (E[:, 0], E[:, 1])), shape=(len(NM), len(NM))), directed=False)
    if cc == 1: break
    main = lab == np.bincount(lab).argmax(); used = set(attach.values()) | set(pattach.values()) | set(E.flatten().tolist())
    for c in range(cc):
        if (lab == c).all() or main[lab == c].any(): continue
        idx = np.where(lab == c)[0]
        if not any(int(i) in used for i in idx): continue                        # orphan nodes nothing refers to: leave
        d = np.linalg.norm(NM[idx][:, None, :] - NM[main][None, :, :], axis=2); i, j = np.unravel_index(int(d.argmin()), d.shape)
        a, b = idx[i], np.where(main)[0][j]
        P, c = reroute(NM[a], NM[b]); P = simplify(P, 1.0); Lr = np.linalg.norm(np.diff(P, axis=0), axis=1).sum()
        if c / max(Lr, 1) < 40 and len(P) > 2 and Lr < 4 * d[i, j] + 40:
            ids = [a]
            for q in P[1:-1]: NM = np.vstack([NM, q]); ids.append(len(NM) - 1)
            ids.append(b); E = np.vstack([E] + [[[u, v]] for u, v in zip(ids[:-1], ids[1:])]); print(f"  joined component of {len(idx)} nodes to main along a {Lr:.0f} m routed path")
        else:
            E = np.vstack([E, [[a, b]]]); print(f"  joined component of {len(idx)} nodes to main with a {d[i, j]:.0f} m link")
        break
    else: break

# ---------- write ----------
NP = inv_exact(NM); err = np.linalg.norm(fwd(NP) - NM, axis=1); print(f"new nodes -> pdf: max forward error {err.max():.2f} m")
G["nodes"] = NP.round(2).tolist(); G["edges"] = [[int(a), int(b)] for a, b in E]; G["roomNode"] = {k: int(v) for k, v in attach.items()}; G["placeNode"] = {k: int(v) for k, v in pattach.items()}
for k, v in rooms.items(): p = inv_exact(ll2m(v[0], v[1]))[0]; v[2] = round(float(p[0]), 2); v[3] = round(float(p[1]), 2)
for k, v in landmarks.items(): p = inv_exact(ll2m(v[0], v[1]))[0]; v[2] = round(float(p[0]), 2); v[3] = round(float(p[1]), 2)
json.dump(D, open(root / args.data, "w"), separators=(",", ":"))
tot = sum(np.linalg.norm(NM[a] - NM[b]) for a, b in E); print(f"wrote {args.data}: {len(NM)} nodes, {len(E)} edges, {tot / 1000:.1f} km ({time.time() - t0:.0f}s)")
