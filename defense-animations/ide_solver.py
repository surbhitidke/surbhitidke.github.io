"""2D Laplace solver for coplanar interdigitated electrodes (IDE) over a conducting liquid.

Finite differences on a uniform node grid. Bottom edge: electrodes are Dirichlet,
gaps are insulating (Neumann). Top edge at height H and both side walls are Neumann.
Two unit cells are supported:
  'pair'     : [G/2 | A | G | B | G/2], walls at the gap centres (the cell described in the brief)
  'periodic' : [W/2 A | G | W/2 B], walls at the finger centres (exact mirror cell of a real IDE)
"""
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spl

V_A, V_B = 0.1, 0.0

def build_cell(W, G, H, d, cell):
    if cell == 'pair':
        edges = [(G/2, G/2 + W, V_A), (G/2 + W + G, G/2 + 2*W + G, V_B)]
        width = 2*W + 2*G
    elif cell == 'periodic':
        edges = [(0, W/2, V_A), (W/2 + G, W + G, V_B)]
        width = W + G
    else:
        raise ValueError(cell)
    nx = int(round(width/d)); ny = int(round(H/d))
    x = np.arange(nx + 1)*d; y = np.arange(ny + 1)*d
    return x, y, edges

def solve(W, G, H, d, cell='pair'):
    x, y, edges = build_cell(W, G, H, d, cell)
    nx, ny = len(x), len(y)
    idx = lambda i, j: j*nx + i
    N = nx*ny
    rows, cols, vals = [], [], []
    b = np.zeros(N)
    fixed = np.full(N, np.nan)
    for (x0, x1, v) in edges:
        for i in range(nx):
            if x0 - 1e-9 <= x[i] <= x1 + 1e-9:
                fixed[idx(i, 0)] = v
    I, J = np.meshgrid(np.arange(nx), np.arange(ny))
    I = I.ravel(); J = J.ravel()
    k = J*nx + I
    free = np.isnan(fixed[k])
    def nb(di, dj):
        ii = I + di; jj = J + dj
        ii = np.where(ii < 0, I - di, ii); ii = np.where(ii > nx-1, I - di, ii)  # mirror (Neumann)
        jj = np.where(jj < 0, J - dj, jj); jj = np.where(jj > ny-1, J - dj, jj)
        return jj*nx + ii
    diag = np.ones(N)
    for di, dj in [(1,0),(-1,0),(0,1),(0,-1)]:
        rows.append(k[free]); cols.append(nb(di, dj)[free]); vals.append(np.full(free.sum(), -1.0))
    diag[free] = 4.0
    b[~np.isnan(fixed)] = fixed[~np.isnan(fixed)]
    A = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(N, N)) + sp.diags(diag)
    phi = spl.spsolve(A.tocsc(), b).reshape(ny, nx)
    return x, y, phi, edges

def field(x, y, phi):
    d = x[1] - x[0]
    Ey, Ex = np.gradient(-phi, d)
    return Ex, Ey, np.hypot(Ex, Ey)

def e_depth(y, prof, ref_y=None):
    """1/e depth of a decaying profile, linear interpolation in log space; ref = value at first sample."""
    ref = prof[0]
    t = ref/np.e
    below = np.where(prof <= t)[0]
    if len(below) == 0:
        return np.nan
    j = below[0]
    y0, y1, p0, p1 = y[j-1], y[j], prof[j-1], prof[j]
    f = (np.log(t) - np.log(p0))/(np.log(p1) - np.log(p0))
    return y0 + f*(y1 - y0)


def solve_grid(nW, ng, nH, d, cell='pair'):
    """Integer-node version: finger width nW nodes, gap ng nodes (even), height nH nodes, spacing d."""
    W = nW*d; G = ng*d; H = nH*d
    return solve(W, G, H, d, cell)
