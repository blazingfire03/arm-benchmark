import numpy as np
from scipy.ndimage import gaussian_filter1d


def interpolate(a, b, max_step=0.05):
    a = np.asarray(a, float); b = np.asarray(b, float)
    dist = float(np.max(np.abs(b - a)))
    n = max(2, int(np.ceil(dist / max_step)) + 1)
    ts = np.linspace(0.0, 1.0, n)[:, None]
    return a[None, :] + (b - a)[None, :] * ts


def segment_collision_free(world, a, b, max_step=0.05):
    return all(world.is_collision_free(q) for q in interpolate(a, b, max_step))


def shortcut(world, path, max_step=0.05):
    path = [np.asarray(p, float) for p in path]
    if len(path) <= 2:
        return path
    out = [path[0]]
    i = 0
    while i < len(path) - 1:
        j = len(path) - 1
        while j > i + 1 and not segment_collision_free(world, path[i], path[j], max_step):
            j -= 1
        out.append(path[j])
        i = j
    return out


def densify(path, max_step=0.05):
    path = [np.asarray(p, float) for p in path]
    if len(path) < 2:
        return np.array(path)
    chunks = []
    for k, (a, b) in enumerate(zip(path[:-1], path[1:])):
        seg = interpolate(a, b, max_step)
        chunks.append(seg if k == 0 else seg[1:])
    return np.vstack(chunks)


def smooth_dense(world, dense, sigma_frac=0.08, passes=5):
    """Gaussian smoothing on a dense (N,7) path.
    sigma = sigma_frac * N, so longer paths get proportionally smoothed.
    Pins start/end. Reverts colliding waypoints after each pass."""
    N = len(dense)
    if N < 4:
        return dense.copy()
    sigma = max(3.0, sigma_frac * N)
    smoothed = dense.copy()
    orig_start = dense[0].copy()
    orig_end = dense[-1].copy()
    for _ in range(passes):
        filtered = gaussian_filter1d(smoothed, sigma=sigma, axis=0, mode='nearest')
        filtered[0] = orig_start
        filtered[-1] = orig_end
        for i in range(1, N - 1):
            if world.is_collision_free(filtered[i]):
                smoothed[i] = filtered[i]
    return smoothed


def retime_minjerk(path, dt=1/240, max_speed=1.0, min_duration=0.5):
    path = np.asarray(path, float)
    if path.ndim == 1:
        path = path[None, :]
    if len(path) < 2:
        return path
    seg = np.diff(path, axis=0)
    seg_len = np.linalg.norm(seg, axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg_len)])
    L = float(cum[-1])
    if L < 1e-9:
        return np.repeat(path[:1], 4, axis=0)
    T = max(min_duration, L / max_speed)
    N = max(4, int(round(T / dt)))
    tau = np.linspace(0.0, 1.0, N)
    s = L * (10 * tau**3 - 15 * tau**4 + 6 * tau**5)
    idx = np.clip(np.searchsorted(cum, s, side="right") - 1, 0, len(seg_len) - 1)
    out = np.empty((N, path.shape[1]))
    for i in range(N):
        k = idx[i]
        denom = seg_len[k] if seg_len[k] > 1e-12 else 1.0
        frac = np.clip((s[i] - cum[k]) / denom, 0.0, 1.0)
        out[i] = path[k] + frac * seg[k]
    return out
