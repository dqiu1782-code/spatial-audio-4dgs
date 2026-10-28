"""Sound-anchored dynamic object layer, shared by training and the viewer.

Scene = 4DGaussians background + object-frame Gaussians with a rigid pose (q_k, T_k) per timestep.
Poses are interpolated (lerp / nlerp) so the layer renders at any continuous t.
"""
import math

import torch


def quat_mul(a, b):
    aw, ax, ay, az = a.unbind(-1)
    bw, bx, by, bz = b.unbind(-1)
    return torch.stack([aw * bw - ax * bx - ay * by - az * bz,
                        aw * bx + ax * bw + ay * bz - az * by,
                        aw * by - ax * bz + ay * bw + az * bx,
                        aw * bz + ax * by - ay * bx + az * bw], -1)


def quat_to_mat(q):
    q = q / q.norm(dim=-1, keepdim=True)
    w, x, y, z = q.unbind(-1)
    return torch.stack([
        torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], -1),
        torch.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], -1),
        torch.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1)], -2)


class DynObj(torch.nn.Module):
    def __init__(self, n=30000, K=80, T_init=None, radius=0.25, n_sh=16, state=None):
        super().__init__()
        P = torch.nn.Parameter
        if state is not None:
            for k in ("xyz", "dc", "opa", "scl", "rot", "T", "q"):
                setattr(self, k, P(state[k].float().cuda()))
            self.n_sh = int(state.get("n_sh", n_sh))
            return
        u = torch.randn(n, 3)
        u = u / u.norm(dim=1, keepdim=True) * radius * torch.rand(n, 1) ** (1 / 3)
        s0 = (4 / 3 * math.pi * radius ** 3 / n) ** (1 / 3)
        self.xyz = P(u.cuda())
        self.dc = P(torch.zeros(n, 1, 3, device="cuda"))
        self.opa = P(torch.full((n, 1), math.log(0.1 / 0.9), device="cuda"))
        self.scl = P(torch.full((n, 3), math.log(s0), device="cuda"))
        self.rot = P(torch.tensor([[1.0, 0, 0, 0]], device="cuda").repeat(n, 1))
        self.T = P(torch.as_tensor(T_init, dtype=torch.float32).cuda().clone())
        self.q = P(torch.tensor([[1.0, 0, 0, 0]], device="cuda").repeat(K, 1))
        self.n_sh = n_sh

    @property
    def K(self):
        return self.T.shape[0]

    def pose(self, t_norm):
        f = min(max(float(t_norm), 0.0), 1.0) * (self.K - 1)
        k0 = min(int(math.floor(f)), self.K - 1)
        k1 = min(k0 + 1, self.K - 1)
        w = f - k0
        T = (1 - w) * self.T[k0] + w * self.T[k1]
        q = (1 - w) * self.q[k0] + w * self.q[k1]
        return T, q / q.norm()

    def at(self, t_norm):
        T, q = self.pose(t_norm)
        R = quat_to_mat(q)
        means = self.xyz @ R.T + T
        rots = quat_mul(q.expand(self.xyz.shape[0], 4), torch.nn.functional.normalize(self.rot, dim=-1))
        n = self.xyz.shape[0]
        shs = torch.cat([self.dc, torch.zeros(n, self.n_sh - 1, 3, device="cuda")], 1)
        return means, shs, torch.sigmoid(self.opa), torch.exp(self.scl), rots

    def centre_traj(self):
        """Opacity-weighted object centre per timestep, (K, 3). Removes the origin/T gauge ambiguity."""
        with torch.no_grad():
            w = torch.sigmoid(self.opa)
            c = (self.xyz * w).sum(0) / w.sum()
            R = quat_to_mat(self.q)
            return (R @ c + self.T).cpu().numpy()

    def state(self, min_opa=0.0):
        keep = (torch.sigmoid(self.opa)[:, 0] > min_opa)
        d = {k: getattr(self, k).detach()[keep].cpu() for k in ("xyz", "dc", "opa", "scl", "rot")}
        d.update(T=self.T.detach().cpu(), q=self.q.detach().cpu(), n_sh=self.n_sh)
        return d


def background(g, t_norm):
    """Background Gaussians at time t (same deformation + activations as gaussian_renderer.render)."""
    with torch.no_grad():
        xyz = g.get_xyz
        tt = torch.full((xyz.shape[0], 1), float(t_norm), device="cuda")
        m, s, r, o, sh = g._deformation(xyz, g._scaling, g._rotation, g._opacity, g.get_features, tt)
        return m, sh, g.opacity_activation(o), g.scaling_activation(s), g.rotation_activation(r)


def rasterize(cam, parts, bg_color, sh_degree):
    from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
    means, shs, opa, scl, rot = [torch.cat(x, 0) for x in zip(*parts)]
    st = GaussianRasterizationSettings(
        image_height=int(cam.image_height), image_width=int(cam.image_width),
        tanfovx=math.tan(cam.FoVx * 0.5), tanfovy=math.tan(cam.FoVy * 0.5),
        bg=bg_color, scale_modifier=1.0, viewmatrix=cam.world_view_transform,
        projmatrix=cam.full_proj_transform, sh_degree=sh_degree, campos=cam.camera_center,
        prefiltered=False, debug=False)
    m2d = torch.zeros_like(means, requires_grad=means.requires_grad)
    out = GaussianRasterizer(st)(means3D=means, means2D=m2d, shs=shs, colors_precomp=None,
                                 opacities=opa, scales=scl, rotations=rot, cov3D_precomp=None)
    return out[0]
