#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""
analyze_matrix.py — Post-processing for gmx_correlation output matrices.

Reads the plain-text matrix written by gmx_correlation (symmetric r(MI) or
raw MI, or asymmetric transfer-entropy) and provides:

  1. Heatmap visualisation (matplotlib)
  2. Network analysis — community detection (leading-eigenvector / Louvain)
     and centrality measures (eigenvector, betweenness, degree)
  3. PyMOL .pml script coloured by community and scaled by EVC

References
----------
- Lange & Grubmüller (2006) Proteins 62:1053  — r(MI) coefficient
- Kraskov et al. (2004) PRE 69:066138          — KSG estimator
- Ince et al. (2017) eLife 6:e18401            — GCMI
- Schreiber (2000) PRL 85:461                  — transfer entropy
- Girvan & Newman (2002) PNAS 99:7821           — community detection
- DyNoPy (Dantu & Pandini 2024) eLife          — network analysis framework

Usage
-----
    python analyze_matrix.py correl.dat --pdb protein.pdb --out results/

    # Transfer-entropy matrix (asymmetric):
    python analyze_matrix.py transfer_entropy.dat --asymmetric --pdb protein.pdb

    # Skip heatmap, only network + PyMOL:
    python analyze_matrix.py correl.dat --no-heatmap --pdb protein.pdb
"""

import argparse
import os
import sys
import warnings
from pathlib import Path

import numpy as np

# ── optional imports with helpful messages ────────────────────────────────────
try:
    import matplotlib
    matplotlib.use("Agg")          # non-interactive backend; change to "TkAgg" for pop-up
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    warnings.warn("matplotlib not found — heatmap will be skipped. pip install matplotlib")

try:
    import igraph as ig
    HAS_IGRAPH = True
except ImportError:
    HAS_IGRAPH = False
    warnings.warn("igraph not found — network analysis will be skipped. pip install igraph")

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False


# ═══════════════════════════════════════════════════════════════════════════════
# I/O
# ═══════════════════════════════════════════════════════════════════════════════

def read_matrix(path: str) -> np.ndarray:
    """Parse the gmx_correlation plain-text matrix.

    Supports the legacy BLITZ++ format::

        N x M [
          v00 v01 ...
        ]

    and plain space-separated square matrices without header.
    """
    path = Path(path)
    if not path.exists():
        sys.exit(f"ERROR: matrix file not found: {path}")

    with open(path) as fh:
        raw = fh.read()

    # Strip the 'N x M [\n...\n]' wrapper if present
    if "[" in raw:
        inner = raw[raw.index("[") + 1 : raw.rindex("]")]
    else:
        inner = raw

    values = [float(v) for v in inner.split() if v not in ("{", "}")]
    n = int(round(len(values) ** 0.5))
    if n * n != len(values):
        sys.exit(f"ERROR: {len(values)} values do not form a square matrix.")

    # gmx_correlation writes column-major (j varies fastest in inner loop)
    mat = np.array(values).reshape(n, n).T
    return mat


def residue_labels(n: int, offset: int = 1) -> list:
    """Return 1-based residue labels [1, 2, ..., n]."""
    return [str(i + offset) for i in range(n)]


# ═══════════════════════════════════════════════════════════════════════════════
# 1. HEATMAP
# ═══════════════════════════════════════════════════════════════════════════════

def plot_heatmap(mat: np.ndarray,
                 out_path: str,
                 title: str = "Correlation matrix",
                 cmap: str = "RdBu_r",
                 vmin: float = None,
                 vmax: float = None,
                 label_stride: int = 10,
                 dpi: int = 150,
                 asymmetric: bool = False) -> None:
    """Save a publication-quality heatmap of *mat* to *out_path*."""
    if not HAS_MPL:
        print("  [heatmap] skipped — matplotlib unavailable")
        return

    n = mat.shape[0]
    tick_pos   = list(range(0, n, label_stride))
    tick_label = [str(i + 1) for i in tick_pos]

    # For symmetric matrices suppress the diagonal sentinel value (2000)
    display = mat.copy()
    if not asymmetric:
        np.fill_diagonal(display, np.nan)

    if vmin is None:
        vmin = np.nanmin(display)
    if vmax is None:
        vmax = np.nanmax(display)

    # Diverging colormap centred at 0 for r(MI); sequential for TE
    if asymmetric:
        cmap = "viridis"
        centre = None
    else:
        centre = max(abs(vmin), abs(vmax))
        vmin, vmax = -centre, centre

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(display, origin="upper", cmap=cmap,
                   vmin=vmin, vmax=vmax, aspect="auto",
                   interpolation="nearest")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("r(MI)" if not asymmetric else "TE (nats)", fontsize=11)

    ax.set_xticks(tick_pos); ax.set_xticklabels(tick_label, fontsize=7, rotation=90)
    ax.set_yticks(tick_pos); ax.set_yticklabels(tick_label, fontsize=7)
    ax.set_xlabel("Residue", fontsize=12)
    ax.set_ylabel("Residue", fontsize=12)
    ax.set_title(title, fontsize=13)

    plt.tight_layout()
    plt.savefig(out_path, dpi=dpi)
    plt.close(fig)
    print(f"  [heatmap] saved → {out_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# 2. NETWORK ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════

def auto_threshold(mat: np.ndarray, asymmetric: bool = False) -> float:
    """Return mean + 0.5*std of off-diagonal values as a sensible default threshold.

    For r(MI) matrices nearly all off-diagonal values are positive, so
    threshold=0 creates a near-complete graph.  This heuristic keeps roughly
    the top ~30% of edges depending on the distribution.
    """
    n = mat.shape[0]
    if asymmetric:
        mask = ~np.eye(n, dtype=bool)
    else:
        # Upper triangle only for symmetric matrices
        mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    vals = mat[mask]
    vals = vals[vals > 0]
    if len(vals) == 0:
        return 0.0
    return float(np.mean(vals) + 0.5 * np.std(vals))


def build_graph(mat: np.ndarray,
                threshold: float = 0.0,
                asymmetric: bool = False) -> "ig.Graph":
    """Construct an igraph graph from the correlation matrix.

    Parameters
    ----------
    mat        : square correlation/TE matrix
    threshold  : only edges with weight > threshold are included
    asymmetric : if True, build a directed graph (for transfer entropy)
    """
    n = mat.shape[0]
    edges, weights = [], []

    if asymmetric:
        # Directed: mat[i,j] = TE(j→i); edge j→i
        for i in range(n):
            for j in range(n):
                if i != j and mat[i, j] > threshold:
                    edges.append((j, i))
                    weights.append(float(mat[i, j]))
        g = ig.Graph(n=n, edges=edges, directed=True)
    else:
        # Symmetric: upper triangle only
        for i in range(n):
            for j in range(i + 1, n):
                if mat[i, j] > threshold:
                    edges.append((i, j))
                    weights.append(float(mat[i, j]))
        g = ig.Graph(n=n, edges=edges, directed=False)

    g.vs["name"]   = [str(i + 1) for i in range(n)]
    g.es["weight"] = weights
    return g


def detect_communities(g: "ig.Graph",
                       method: str = "leading_eigenvector",
                       min_size: int = 3) -> "ig.VertexClustering":
    """Run community detection and return a VertexClustering.

    Parameters
    ----------
    g        : undirected igraph Graph (directed graphs are symmetrised first)
    method   : 'leading_eigenvector' (default, as in DyNoPy) or 'louvain'
    min_size : communities smaller than this are merged into a noise community
    """
    # Community detection works on undirected graphs
    ug = g.as_undirected(combine_edges="max") if g.is_directed() else g

    if method == "leading_eigenvector":
        cl = ug.community_leading_eigenvector(weights=ug.es["weight"]
                                               if ug.ecount() > 0 else None)
    elif method == "louvain":
        cl = ug.community_multilevel(weights=ug.es["weight"]
                                      if ug.ecount() > 0 else None)
    else:
        sys.exit(f"Unknown community method: {method}")

    # Relabel: tiny communities → community 0 (noise)
    membership = list(cl.membership)
    sizes = {}
    for m in membership:
        sizes[m] = sizes.get(m, 0) + 1

    remap = {}
    next_id = 1
    for comm_id, sz in sorted(sizes.items()):
        if sz >= min_size:
            remap[comm_id] = next_id
            next_id += 1
        else:
            remap[comm_id] = 0   # noise

    membership = [remap[m] for m in membership]
    n_communities = next_id - 1
    print(f"  [network] {n_communities} communities ≥ {min_size} residues "
          f"(Q = {cl.modularity:.3f})")
    return ig.VertexClustering(ug, membership), cl.modularity


def compute_centrality(g: "ig.Graph") -> dict:
    """Compute eigenvector centrality, betweenness, and degree.

    Returns a dict of arrays indexed by vertex id.
    """
    ug = g.as_undirected(combine_edges="max") if g.is_directed() else g
    weights = ug.es["weight"] if ug.ecount() > 0 else None

    evc   = np.array(ug.eigenvector_centrality(weights=weights, directed=False))
    btwn  = np.array(ug.betweenness(weights=weights, directed=False))
    # Normalise betweenness to [0,1]
    bmax  = btwn.max() if btwn.max() > 0 else 1.0
    btwn /= bmax
    deg   = np.array(ug.degree())

    return {"evc": evc, "betweenness": btwn, "degree": deg}


def network_summary(g: "ig.Graph",
                    membership: list,
                    centrality: dict,
                    out_csv: str) -> None:
    """Write per-residue network statistics to a CSV file."""
    n = g.vcount()
    rows = []
    for i in range(n):
        rows.append({
            "residue":     int(g.vs[i]["name"]),
            "community":   membership[i],
            "evc":         centrality["evc"][i],
            "betweenness": centrality["betweenness"][i],
            "degree":      centrality["degree"][i],
        })

    if HAS_PANDAS:
        import pandas as pd
        df = pd.DataFrame(rows)
        df.sort_values("evc", ascending=False, inplace=True)
        df.to_csv(out_csv, index=False, float_format="%.5f")
    else:
        with open(out_csv, "w") as fh:
            fh.write("residue,community,evc,betweenness,degree\n")
            for r in sorted(rows, key=lambda x: -x["evc"]):
                fh.write(f"{r['residue']},{r['community']},"
                         f"{r['evc']:.5f},{r['betweenness']:.5f},{r['degree']}\n")
    print(f"  [network] residue stats saved → {out_csv}")


def plot_network(g: "ig.Graph",
                 membership: list,
                 centrality: dict,
                 out_path: str,
                 max_edges: int = 500) -> None:
    """Save a 2-D spring-layout network diagram coloured by community."""
    if not HAS_MPL:
        return

    n      = g.vcount()
    n_comm = max(membership) + 1
    palette = plt.cm.get_cmap("tab20", max(n_comm, 1))
    node_color = [palette(m) if m > 0 else (0.7, 0.7, 0.7, 1.0)
                  for m in membership]
    evc = centrality["evc"]
    node_size = 20 + 200 * evc   # scale node size by EVC

    # Spring layout via igraph
    ug = g.as_undirected(combine_edges="max") if g.is_directed() else g
    try:
        layout = ug.layout("fr")   # Fruchterman-Reingold
    except Exception:
        layout = ug.layout("kk")

    xs = [layout[i][0] for i in range(n)]
    ys = [layout[i][1] for i in range(n)]

    fig, ax = plt.subplots(figsize=(9, 8))

    # Draw edges (subsample if graph is dense)
    edges = [(e.source, e.target) for e in ug.es]
    if len(edges) > max_edges:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(edges), max_edges, replace=False)
        edges = [edges[i] for i in idx]

    for src, tgt in edges:
        ax.plot([xs[src], xs[tgt]], [ys[src], ys[tgt]],
                color="lightgray", linewidth=0.4, zorder=1)

    sc = ax.scatter(xs, ys, s=node_size, c=node_color, zorder=2,
                    edgecolors="white", linewidths=0.3)

    # Label residues with top EVC
    top_n = min(20, n)
    top_idx = np.argsort(evc)[::-1][:top_n]
    for i in top_idx:
        ax.annotate(g.vs[i]["name"], (xs[i], ys[i]),
                    fontsize=5, ha="center", va="bottom",
                    color="black", zorder=3)

    ax.set_axis_off()
    ax.set_title("Residue correlation network (node size ∝ EVC)", fontsize=12)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  [network] graph diagram saved → {out_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# 3. PyMOL .pml OUTPUT
# ═══════════════════════════════════════════════════════════════════════════════

# Community colours: community 0 = noise (grey), 1..N use the palette below.
# Extend as needed; cycles if more communities than colours.
_COMMUNITY_COLOURS = [
    (0.75, 0.75, 0.75),   # 0 — noise / unassigned
    (0.85, 0.10, 0.10),   # 1 — red (core)
    (0.20, 0.40, 0.80),   # 2 — blue
    (0.10, 0.70, 0.10),   # 3 — green
    (0.90, 0.55, 0.00),   # 4 — orange
    (0.55, 0.00, 0.80),   # 5 — purple
    (0.00, 0.75, 0.75),   # 6 — cyan
    (0.90, 0.90, 0.00),   # 7 — yellow
    (0.90, 0.40, 0.60),   # 8 — pink
    (0.40, 0.20, 0.10),   # 9 — brown
    (0.50, 0.80, 0.20),   # 10 — lime
]


def _comm_colour(comm_id: int) -> tuple:
    if comm_id == 0:
        return _COMMUNITY_COLOURS[0]
    idx = 1 + ((comm_id - 1) % (len(_COMMUNITY_COLOURS) - 1))
    return _COMMUNITY_COLOURS[idx]


def write_pml(g: "ig.Graph",
              membership: list,
              centrality: dict,
              pdb_path: str,
              out_pml: str,
              sphere_scale_max: float = 1.5,
              edge_threshold_pct: float = 0.90,
              chain: str = "A") -> None:
    """Write a PyMOL script that colours residues by community and EVC.

    The script:
    - Loads the PDB (if *pdb_path* is provided)
    - Shows the protein as cartoon, community residues as coloured spheres
    - Scales sphere size by eigenvector centrality
    - Draws CGO cylinders between the top-weighted network edges
    - Sets a white background and ray-traces a PNG snapshot

    Parameters
    ----------
    g                 : igraph Graph
    membership        : community id per vertex (0 = noise)
    centrality        : dict with 'evc' array
    pdb_path          : path to PDB file, or "" to skip load command
    out_pml           : output .pml file path
    sphere_scale_max  : maximum sphere_scale for highest-EVC residue
    edge_threshold_pct: draw edges with weight ≥ this percentile of all weights
    chain             : chain identifier in the PDB
    """
    n   = g.vcount()
    evc = centrality["evc"]
    evc_max = evc.max() if evc.max() > 0 else 1.0

    # Collect all edge weights to compute percentile threshold
    ug = g.as_undirected(combine_edges="max") if g.is_directed() else g
    all_weights = np.array(ug.es["weight"]) if ug.ecount() > 0 else np.array([])
    if len(all_weights) > 0:
        edge_wt_cutoff = np.percentile(all_weights, edge_threshold_pct * 100)
    else:
        edge_wt_cutoff = 0.0

    lines = []
    pml = lines.append   # convenience alias

    pml("# ── gmx_correlation network visualisation ─────────────────────────")
    pml(f"# Generated by analyze_matrix.py")
    pml(f"# Communities: {max(membership)} | Residues in graph: {n}")
    pml("")

    # ── Load structure ────────────────────────────────────────────────────────
    if pdb_path:
        pdb_abs = str(Path(pdb_path).resolve())
        pml(f'load {pdb_abs}, protein')
    else:
        pml("# No PDB provided — load your structure manually: load protein.pdb, protein")
        pml("# Adjust 'protein' selection name below if needed")
        pml("cmd.create('protein', 'all')")

    pml("")
    pml("# ── Base representation ──────────────────────────────────────────────")
    pml("hide everything, protein")
    pml("show cartoon,   protein")
    pml("color grey80,   protein")
    pml("set cartoon_transparency, 0.4, protein")
    pml("")

    # ── Per-residue spheres ───────────────────────────────────────────────────
    pml("# ── Community spheres (CA atoms) ────────────────────────────────────")

    # Group residues by community for cleaner PyMOL selections
    comm_residues: dict[int, list] = {}
    for i in range(n):
        cid = membership[i]
        comm_residues.setdefault(cid, []).append(int(g.vs[i]["name"]))

    for cid in sorted(comm_residues):
        rlist   = comm_residues[cid]
        r, g_c, b = _comm_colour(cid)
        sel_name = f"comm_{cid}" if cid > 0 else "noise"
        res_str  = "+".join(str(r) for r in rlist)
        pml(f"select {sel_name}, (protein and chain {chain} and resi {res_str} and name CA)")
        pml(f"show spheres, {sel_name}")
        pml(f"color [{int(r*255)},{int(g_c*255)},{int(b*255)}], {sel_name}")

    pml("")
    pml("# ── Sphere sizes scaled by eigenvector centrality ──────────────────")
    for i in range(n):
        resid   = int(g.vs[i]["name"])
        scale   = 0.3 + sphere_scale_max * float(evc[i]) / evc_max
        pml(f"set sphere_scale, {scale:.3f}, (protein and chain {chain} and resi {resid} and name CA)")

    pml("")

    # ── Network edges as CGO cylinders ────────────────────────────────────────
    # Build the edge list in Python (not per-edge PyMOL commands) so that the
    # embedded script is O(1) lines regardless of network size.
    pml("# ── Network edges (top-weighted pairs as CGO cylinders) ─────────────")

    # Collect edge data as a compact list of tuples written once into the .pml
    edge_tuples = []
    wmax = all_weights.max() if len(all_weights) > 0 else 1.0
    for e in ug.es:
        if e["weight"] < edge_wt_cutoff:
            continue
        src = int(ug.vs[e.source]["name"])
        tgt = int(ug.vs[e.target]["name"])
        r1, g1, b1 = _comm_colour(membership[e.source])
        r2, g2, b2 = _comm_colour(membership[e.target])
        w = float(e["weight"])
        w_norm = (w - edge_wt_cutoff) / max(wmax - edge_wt_cutoff, 1e-9)
        radius = 0.05 + 0.15 * w_norm
        edge_tuples.append((src, tgt, radius, r1, g1, b1, r2, g2, b2))

    pml("python")
    pml("from pymol import cmd")
    pml("from chempy.cgo import CYLINDER")
    pml("")
    pml("# Edge list: (resi_src, resi_tgt, radius, r1,g1,b1, r2,g2,b2)")
    pml(f"_EDGES = {edge_tuples!r}")
    pml(f"_CHAIN = {chain!r}")
    pml(f"_SEL   = 'protein'")
    pml("")
    pml("def draw_edges():")
    pml("    obj = []")
    pml("    for src, tgt, rad, r1,g1,b1, r2,g2,b2 in _EDGES:")
    pml("        s1 = f'{_SEL} and chain {_CHAIN} and resi {src} and name CA'")
    pml("        s2 = f'{_SEL} and chain {_CHAIN} and resi {tgt} and name CA'")
    pml("        p1 = cmd.get_atom_coords(s1)")
    pml("        p2 = cmd.get_atom_coords(s2)")
    pml("        if p1 and p2:")
    pml("            obj += [CYLINDER,")
    pml("                    p1[0],p1[1],p1[2], p2[0],p2[1],p2[2],")
    pml("                    rad, r1,g1,b1, r2,g2,b2]")
    pml("    if obj:")
    pml("        cmd.load_cgo(obj, 'network_edges')")
    pml("")
    pml("draw_edges()")
    pml("python end")
    pml("")

    # ── Display settings ──────────────────────────────────────────────────────
    pml("# ── Display settings ─────────────────────────────────────────────────")
    pml("bg_color white")
    pml("set ray_opaque_background, 1")
    pml("set sphere_transparency, 0.0")
    pml("set depth_cue, 0")
    pml("set antialias, 2")
    pml("orient protein")
    pml("zoom protein, 5")
    pml("")

    # ── Legend labels ─────────────────────────────────────────────────────────
    pml("# ── Community legend (pseudo-atoms at origin) ────────────────────────")
    for cid in sorted(comm_residues):
        if cid == 0:
            continue
        rlist = comm_residues[cid]
        r, g_c, b = _comm_colour(cid)
        label = f"Community {cid} ({len(rlist)} residues)"
        pml(f"pseudoatom legend_comm{cid}, pos=[999,{cid*2},0], label='{label}'")
        pml(f"color [{int(r*255)},{int(g_c*255)},{int(b*255)}], legend_comm{cid}")
    pml("hide everything, legend_comm*")
    pml("")

    pml("# ── Optional: save a PNG ─────────────────────────────────────────────")
    pml("# ray 1200, 900")
    pml("# png network_view.png, dpi=300")
    pml("")
    pml("print('Network loaded. Communities: "
        + str(max(membership) if membership else 0) + "')")

    with open(out_pml, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"  [pymol]   script saved  → {out_pml}")
    print(f"            Run: pymol {out_pml}")


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="Analyse gmx_correlation output matrices: heatmap, network, PyMOL")
    p.add_argument("matrix", help="Matrix file written by gmx_correlation")
    p.add_argument("--pdb",        default="", help="PDB file for PyMOL visualisation")
    p.add_argument("--out",        default=".", help="Output directory (default: current dir)")
    p.add_argument("--prefix",     default="",  help="Filename prefix (default: matrix basename)")
    p.add_argument("--threshold",  type=float, default=None,
                   help="Edge weight threshold for graph (default: auto = mean + 0.5*std "
                        "of off-diagonal values; pass 0 to include all positive edges)")
    p.add_argument("--method",     default="leading_eigenvector",
                   choices=["leading_eigenvector", "louvain"],
                   help="Community detection algorithm (default: leading_eigenvector)")
    p.add_argument("--min-community", type=int, default=3,
                   help="Minimum community size (default 3)")
    p.add_argument("--chain",      default="A", help="PDB chain for PyMOL (default A)")
    p.add_argument("--edge-pct",   type=float, default=0.90,
                   help="Percentile of edge weights shown as CGO cylinders (default 0.90 = top 10%%)")
    p.add_argument("--asymmetric", action="store_true",
                   help="Matrix is asymmetric (transfer entropy). Builds directed graph.")
    p.add_argument("--no-heatmap", action="store_true", help="Skip heatmap")
    p.add_argument("--no-network", action="store_true", help="Skip network analysis")
    p.add_argument("--no-pymol",   action="store_true", help="Skip .pml output")
    p.add_argument("--label-stride", type=int, default=10,
                   help="Heatmap axis tick stride (default 10)")
    p.add_argument("--cmap",       default="RdBu_r",
                   help="Matplotlib colormap for heatmap (default RdBu_r)")
    p.add_argument("--residue-offset", type=int, default=1,
                   help="First residue number (default 1)")
    return p.parse_args()


def main():
    args = parse_args()

    # ── Setup ─────────────────────────────────────────────────────────────────
    mat_path = Path(args.matrix)
    out_dir  = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    prefix = args.prefix or mat_path.stem

    print(f"\n{'═'*60}")
    print(f"  gmx_correlation matrix analysis")
    print(f"  Input : {mat_path}")
    print(f"  Output: {out_dir}/")
    print(f"{'═'*60}")

    # ── Read matrix ────────────────────────────────────────────────────────────
    print("\n[1/3] Reading matrix …")
    mat = read_matrix(str(mat_path))
    n   = mat.shape[0]
    print(f"  Shape: {n} × {n}  |  min={mat.min():.4f}  max={mat.max():.4f}")

    # ── Heatmap ────────────────────────────────────────────────────────────────
    if not args.no_heatmap:
        print("\n[2/3] Heatmap …")
        if HAS_MPL:
            title = f"{prefix}  ({n} residues)"
            plot_heatmap(mat,
                         out_path=str(out_dir / f"{prefix}_heatmap.png"),
                         title=title,
                         cmap=args.cmap,
                         label_stride=args.label_stride,
                         asymmetric=args.asymmetric)
        else:
            print("  [heatmap] skipped")

    # ── Network analysis ───────────────────────────────────────────────────────
    if not args.no_network:
        print("\n[3/3] Network analysis …")
        if not HAS_IGRAPH:
            print("  [network] skipped — igraph not installed")
        else:
            threshold = args.threshold
            if threshold is None:
                threshold = auto_threshold(mat, asymmetric=args.asymmetric)
                print(f"  Auto threshold: {threshold:.4f}  (mean + 0.5·std of positive off-diag values)")
            elif threshold == 0.0:
                print("  WARNING: threshold=0 includes all positive edges — graph may be near-complete")

            g = build_graph(mat, threshold=threshold,
                            asymmetric=args.asymmetric)
            print(f"  Graph: {g.vcount()} nodes, {g.ecount()} edges "
                  f"(threshold > {threshold:.4f})")

            if g.ecount() == 0:
                print("  WARNING: no edges above threshold — try lowering --threshold")
                return

            # Community detection (undirected)
            cl, modularity = detect_communities(
                g, method=args.method, min_size=args.min_community)
            membership = cl.membership

            # Centrality
            centrality = compute_centrality(g)

            # Save CSV
            network_summary(g, membership, centrality,
                            out_csv=str(out_dir / f"{prefix}_network.csv"))

            # Network diagram
            if HAS_MPL:
                plot_network(g, membership, centrality,
                             out_path=str(out_dir / f"{prefix}_network.png"))

            # PyMOL
            if not args.no_pymol:
                write_pml(g, membership, centrality,
                          pdb_path=args.pdb,
                          out_pml=str(out_dir / f"{prefix}_network.pml"),
                          edge_threshold_pct=args.edge_pct,
                          chain=args.chain)

    print(f"\n{'═'*60}")
    print(f"  Done.  Results in {out_dir}/")
    print(f"{'═'*60}\n")


if __name__ == "__main__":
    main()
