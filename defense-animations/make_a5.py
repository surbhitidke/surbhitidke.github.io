"""A5: field penetration of three IDE geometries. Every frame is a fresh finite-difference
solution of Laplace's equation (no pixel morphing). Geometry morphs B -> A -> C in log space.

Usage: python3 make_a5.py [--still] [--workers N]
"""
import os, sys, shutil, subprocess, multiprocessing as mp
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, LogNorm
from matplotlib.patches import Rectangle
import matplotlib.patheffects as pe
import ide_solver as S

OUT = os.path.dirname(os.path.abspath(__file__))
FPS, NG, DV = 30, 32, 0.1
NAVY, SKY, ICE, INK, MUTED, GOLD, GRID = '#0F2C59', '#0284C7', '#E0F2FE', '#0F172A', '#475569', '#C9A227', '#CBD5E1'
KEYS = {'B': dict(W=20., G=5., H=100., maxwell=5.9),
        'A': dict(W=120., G=20., H=100., maxwell=24.4),
        'C': dict(W=300., G=150., H=600., maxwell=174.8)}
# timeline (seconds): hold B, morph B->A, hold A, morph A->C, hold C = 8 s
SEG = [('hold', 'B', 1.5), ('morph', ('B', 'A'), 1.5), ('hold', 'A', 1.5), ('morph', ('A', 'C'), 1.5), ('hold', 'C', 2.0)]
cmap = LinearSegmentedColormap.from_list('ide', [ICE, '#7DD3FC', SKY, NAVY])

def solve_geom(W, G, H):
    d = G/NG
    nW, nH = int(round(W/d)), int(round(H/d))
    x, y, phi, ed = S.solve(nW*d, G, nH*d, d, 'pair')
    Ex, Ey, E = S.field(x, y, phi)
    xc = (ed[0][1] + ed[1][0])/2
    i = int(np.argmin(abs(x - xc)))
    prof = E[:, i]
    depth = S.e_depth(y, prof)
    return dict(x=x, y=y, Ex=Ex, Ey=Ey, E=E, ed=ed, xc=xc, prof=prof/prof[0], depth=depth, W=nW*d, G=G, H=nH*d)

def smooth(t): return t*t*(3 - 2*t)

def frame_plan():
    plan = []
    for kind, what, dur in SEG:
        n = int(round(dur*FPS))
        for k in range(n):
            if kind == 'hold': plan.append(('hold', what, what, 0.0, k == 0))
            else: plan.append(('morph', what[0], what[1], smooth((k + 1)/n), True))
    return plan

def geom_at(a, b, s):
    A, B = KEYS[a], KEYS[b]
    f = lambda k: float(np.exp((1 - s)*np.log(A[k]) + s*np.log(B[k])))
    return f('W'), f('G'), f('H')

REF = {}
def render(args):
    n, (kind, a, b, s, fresh), path = args
    if kind == 'hold': W, G, H = KEYS[a]['W'], KEYS[a]['G'], KEYS[a]['H']; r = REF[a]
    else: W, G, H = geom_at(a, b, s); r = solve_geom(W, G, H)
    fig = plt.figure(figsize=(16, 9), dpi=100, facecolor='white')
    fig.text(0.04, 0.945, 'How deep does the field reach into the liquid?', fontsize=26, color=NAVY, weight='bold', va='center')
    fig.text(0.04, 0.895, 'Coplanar interdigitated electrodes, 2D unit cell, 0.1 V on one finger and 0 V on the other', fontsize=15, color=MUTED, va='center')
    # ---- field map
    x, y = r['x'], r['y']; width = x[-1]
    ymax = min(r['H'], width/1.667)
    pxw, left = 960., 90.
    ppu = pxw/width                                  # pixels per micrometre
    bar = 30/ppu                                     # electrode bar height in data units
    axh = (ymax + bar)*ppu
    base = 150 + (610 - axh)/2
    ax = fig.add_axes([left/1600, base/900, pxw/1600, axh/900])
    jm = np.searchsorted(y, ymax) + 1
    En = r['E'][:jm]/(DV/r['G'])
    ax.imshow(En, origin='lower', extent=[x[0], x[-1], y[0], y[jm-1]], cmap=cmap, norm=LogNorm(0.02, 3), aspect='auto', interpolation='bilinear', zorder=1)
    ax.streamplot(x, y[:jm], r['Ex'][:jm], r['Ey'][:jm], color=INK, linewidth=0.8, density=(1.5, 1.1), arrowsize=0.9, zorder=2)
    for (x0, x1, v) in r['ed']:
        ax.add_patch(Rectangle((x0, -bar), x1 - x0, bar, fc=GOLD, ec=INK, lw=1.0, zorder=3, clip_on=False))
        ax.text((x0 + x1)/2, -bar/2, '0.1 V' if v > 0 else '0 V', ha='center', va='center', fontsize=13, color=INK, weight='bold', zorder=4)
    for u0, u1 in ((0, r['ed'][0][0]), (r['ed'][0][1], r['ed'][1][0]), (r['ed'][1][1], width)):
        ax.add_patch(Rectangle((u0, -bar), u1 - u0, bar, fc=GRID, ec=INK, lw=1.0, zorder=3, clip_on=False))
    ax.set_xlim(0, width); ax.set_ylim(-bar, ymax)
    ax.set_xlabel('Position across the unit cell (µm)', fontsize=14, color=INK)
    ax.set_ylabel('Height above substrate (µm)', fontsize=14, color=INK)
    ax.tick_params(labelsize=13, colors=INK)
    for sp_ in ('top', 'right'): ax.spines[sp_].set_visible(False)
    if ymax >= r['H'] - 1e-6:
        ax.axhline(r['H'], color=INK, lw=2, zorder=3)
        ax.text(width*0.01, r['H'], 'insulating top wall', fontsize=12, color=INK, va='bottom')
    # depth marker on the gap-centre line
    dm = r['depth']
    ha = dict(color='white', lw=4.5, zorder=5)
    ax.plot([r['xc']]*2, [0, dm], **ha); ax.plot([r['xc']]*2, [0, dm], color=INK, lw=2, ls=(0, (4, 2)), zorder=6)
    ax.plot([r['xc']], [dm], 'o', ms=10, mfc='white', mec=INK, mew=2.2, zorder=7)
    ax.annotate(f'1/e depth  {dm:.1f} µm', (r['xc'], dm), xytext=(18, 6), textcoords='offset points', fontsize=16, weight='bold', color=INK, zorder=8,
                bbox=dict(boxstyle='round,pad=0.3', fc='white', ec=INK, lw=1.2), annotation_clip=False)
    # colour bar
    cax = fig.add_axes([left/1600, 62/900, pxw/1600, 0.012])
    sm = plt.cm.ScalarMappable(norm=LogNorm(0.02, 3), cmap=cmap)
    cb = fig.colorbar(sm, cax=cax, orientation='horizontal', ticks=[0.02, 0.1, 1, 3]); cb.ax.set_xticklabels(['0.02', '0.1', '1', '3'])
    cb.ax.tick_params(labelsize=12, colors=INK); cb.outline.set_visible(False)
    cb.set_label('|E| in units of ΔV/G (log scale)', fontsize=13, color=INK, labelpad=2)
    # ---- right panel: normalised |E| along the gap-centre line
    rx0 = 1160
    ax2 = fig.add_axes([rx0/1600, 500/900, 390/1600, 250/900])
    for k, c in zip('BAC', ('#7DD3FC', SKY, NAVY)):
        q = REF[k]; m = q['y'] >= 1
        ax2.plot(q['y'][m], q['prof'][m], color=c, lw=1.6, alpha=0.45, zorder=1)
        ax2.plot([q['depth']], [1/np.e], 'o', ms=6, color=c, alpha=0.7, zorder=2)
        ax2.text(q['depth']*1.12, 1/np.e + 0.08, k, fontsize=13, color=MUTED, weight='bold')
    ax2.axhline(1/np.e, color=MUTED, lw=1, ls=':', zorder=0)
    m = r['y'] >= 1
    ax2.plot(r['y'][m], r['prof'][m], color=INK, lw=2.6, zorder=4)
    ax2.plot([dm], [1/np.e], 'o', ms=10, mfc='white', mec=INK, mew=2.2, zorder=5)
    ax2.set_xscale('log'); ax2.set_xlim(1, 700); ax2.set_ylim(0, 1.05)
    ax2.set_xlabel('Height above the gap centre (µm)', fontsize=13, color=INK)
    ax2.set_ylabel('|E| / |E| at the surface', fontsize=13, color=INK)
    ax2.tick_params(labelsize=12, colors=INK)
    ax2.text(1.2, 1/np.e + 0.02, '1/e', fontsize=12, color=MUTED)
    for sp_ in ('top', 'right'): ax2.spines[sp_].set_visible(False)
    fig.text(rx0/1600, 0.88, 'Field strength straight above the gap', fontsize=15, color=INK, weight='bold', va='center')
    # ---- readout
    name = a if kind == 'hold' else None
    title = f'Design {name}' if name else f'Morphing  {a} → {b}'
    fig.text(rx0/1600, 380/900, title, fontsize=22, color=NAVY, weight='bold', va='center')
    fig.text(rx0/1600, 340/900, f'finger W = {W:.3g} µm     gap G = {G:.3g} µm', fontsize=15, color=INK, va='center')
    fig.text(rx0/1600, 310/900, f'cell height H = {H:.3g} µm', fontsize=15, color=INK, va='center')
    fig.text(rx0/1600, 262/900, f'1/e depth (this solver):  {dm:.1f} µm', fontsize=17, color=INK, weight='bold', va='center')
    if name:
        mw = KEYS[name]['maxwell']
        fig.text(rx0/1600, 228/900, f'1/e depth (Maxwell, chapter):  {mw:.1f} µm', fontsize=17, color=INK, va='center')
        fig.text(rx0/1600, 196/900, f'difference  {100*(dm - mw)/mw:+.1f} %', fontsize=15, color=MUTED, va='center')
    else:
        fig.text(rx0/1600, 228/900, 'In-between geometry, solved at every frame', fontsize=14, color=MUTED, va='center')
    fig.text(rx0/1600, 130/900, 'Finite-difference solution of Laplace\'s equation.\nInsulating gaps, side walls and top wall.\nDepth is where |E| above the gap centre\nfalls to 1/e of its surface value.', fontsize=12, color=MUTED, va='top', linespacing=1.4)
    fig.savefig(path, dpi=100, facecolor='white')
    plt.close(fig)
    return n

def main():
    still = '--still' in sys.argv
    workers = int(sys.argv[sys.argv.index('--workers') + 1]) if '--workers' in sys.argv else 4
    for k, v in KEYS.items(): REF[k] = solve_geom(v['W'], v['G'], v['H'])
    for k in 'BAC': print(k, f"depth {REF[k]['depth']:.2f} um (chapter Maxwell {KEYS[k]['maxwell']})")
    plan = frame_plan()
    fd = os.path.join(OUT, '_frames'); shutil.rmtree(fd, ignore_errors=True); os.makedirs(fd)
    if still:
        for k in 'BAC': render((0, ('hold', k, k, 0.0, True), os.path.join(OUT, f'still_{k}.png')))
        render((0, ('morph', 'A', 'C', 0.5, True), os.path.join(OUT, 'still_AC_mid.png')))
        return
    jobs, firsts = [], {}
    for n, p in enumerate(plan):
        path = os.path.join(fd, f'f{n:04d}.png')
        if p[0] == 'hold' and p[1] in firsts: continue          # hold frames are identical, copy later
        if p[0] == 'hold': firsts[p[1]] = path
        jobs.append((n, p, path))
    with mp.get_context('fork').Pool(workers) as pool:
        for i, _ in enumerate(pool.imap_unordered(render, jobs)):
            if i % 10 == 0: print('rendered', i, '/', len(jobs), flush=True)
    for n, p in enumerate(plan):
        path = os.path.join(fd, f'f{n:04d}.png')
        if not os.path.exists(path): shutil.copy(firsts[p[1]], path)
    mp4 = os.path.join(OUT, 'A5_ide_field_penetration.mp4')
    subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-framerate', str(FPS), '-i', os.path.join(fd, 'f%04d.png'),
                    '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '16', '-movflags', '+faststart', mp4], check=True)
    gif = os.path.join(OUT, 'A5_ide_field_penetration.gif')
    vf = 'fps=15,scale=1600:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors=128:stats_mode=diff[p];[b][p]paletteuse=dither=none'
    subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-i', mp4, '-vf', vf, '-loop', '0', gif], check=True)
    shutil.rmtree(fd)
    print('done', mp4, gif)

if __name__ == '__main__':
    main()
