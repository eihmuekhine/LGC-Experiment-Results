"""Attack spaces (pixel / latent-RAG) and the counted hard-label oracle."""

from __future__ import annotations

from collections import Counter
from typing import Optional, Tuple

import torch
import torch.nn as nn

from .env import empty_cache, is_oom


class AttackSpace:
    """Coordinates in which an attack moves. ``origin`` has shape (1, *shape)."""

    name: str = "abstract"
    is_latent: bool = False

    def __init__(self, origin: torch.Tensor, x0: torch.Tensor):
        self.origin = origin.detach()
        self.x0 = x0.detach()

    @property
    def shape(self) -> tuple:
        return tuple(self.origin.shape[1:])

    @property
    def dim(self) -> int:
        return int(self.origin.numel())

    def project(self, z: torch.Tensor) -> torch.Tensor:
        """Map an arbitrary point to the feasible set of the space."""
        return z

    @torch.no_grad()
    def to_image(self, z: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class PixelSpace(AttackSpace):
    name = "pixel"
    is_latent = False

    def __init__(self, x0: torch.Tensor):
        super().__init__(origin=x0, x0=x0)

    def project(self, z: torch.Tensor) -> torch.Tensor:
        return torch.clamp(z, 0.0, 1.0)

    @torch.no_grad()
    def to_image(self, z: torch.Tensor) -> torch.Tensor:
        return torch.clamp(z, 0.0, 1.0)


class LatentRAGSpace(AttackSpace):
    """Residual-based Adversarial Generation: x(z) = clip(x0 + G(z) - G(z_s), 0, 1)."""

    name = "latent"
    is_latent = True

    def __init__(self, z_s: torch.Tensor, x0: torch.Tensor, decoder: nn.Module):
        super().__init__(origin=z_s, x0=x0)
        self.decoder = decoder
        with torch.no_grad():
            self.x_rec0 = decoder(z_s).detach()

    @torch.no_grad()
    def to_image(self, z: torch.Tensor) -> torch.Tensor:
        return torch.clamp(self.x0 + self.decoder(z) - self.x_rec0, 0.0, 1.0)


class LatentResidualSpace(AttackSpace):
    """RAG latent space plus one residual coordinate (hard-label targeted start, final LGC methods).

    Coordinates w = (z, u), flattened to shape (1, k + 1) (non-targeted: (1, k), plain RAG):
        x(z, u) = clip(x0 + G(z) - G(z_s) + (u / lam) (r_t - r_s), 0, 1),
        r = x - G(E(x)) (autoencoder residual),  origin = (z_s, 0) -> x0.
    With z_t = E(x_t) and u = lam the image is EXACTLY x_t:
        x0 + G(z_t) - G(z_s) + (x_t - G(z_t)) - (x0 - G(z_s)) = x_t,
    so the target class is reached without any victim gradient (the original targeted
    initialisation ran Adam on the victim's cross-entropy, which violates the hard-label threat
    model). lam = ||r_t - r_s|| / g, g = ||G(z_t) - G(z_s)|| / ||z_t - z_s|| (mean decoder gain along
    the chord), gives the residual coordinate the same pixel change per unit as a latent coordinate,
    so the Euclidean objective in w does not favour either part.
    """

    name = "latent_r"
    is_latent = True

    def __init__(self, z_s: torch.Tensor, x0: torch.Tensor, decoder: nn.Module,
                 encoder: Optional[nn.Module] = None, x_t: Optional[torch.Tensor] = None):
        self.decoder = decoder
        self.z_shape = tuple(z_s.shape[1:])
        self.k = int(z_s[0].numel())
        with torch.no_grad():
            self.x_rec0 = decoder(z_s).detach()
            self.has_residual = x_t is not None and encoder is not None
            if self.has_residual:
                z_t = encoder(x_t).detach()
                r_s = x0 - self.x_rec0
                r_t = x_t - decoder(z_t)
                self.delta_r = (r_t - r_s).detach()
                dz = float((z_t - z_s).flatten().norm())
                gain = float((decoder(z_t) - self.x_rec0).flatten().norm()) / dz if dz > 1e-12 else 1.0
                self.lam = max(float(self.delta_r.flatten().norm()) / max(gain, 1e-12), 1e-6)
                origin = torch.cat([z_s.reshape(1, -1), torch.zeros(1, 1, device=z_s.device, dtype=z_s.dtype)], 1)
                self.target_point = torch.cat(
                    [z_t.reshape(1, -1), torch.full((1, 1), self.lam, device=z_s.device, dtype=z_s.dtype)], 1)
            else:
                self.delta_r, self.lam, self.target_point = None, 1.0, None
                origin = z_s.reshape(1, -1)
        super().__init__(origin=origin, x0=x0)

    def split(self, w: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        z = w[:, : self.k].reshape(w.shape[0], *self.z_shape)
        u = w[:, self.k] if self.has_residual else None
        return z, u

    @torch.no_grad()
    def to_image(self, w: torch.Tensor) -> torch.Tensor:
        z, u = self.split(w)
        img = self.x0 + self.decoder(z) - self.x_rec0
        if u is not None:
            img = img + (u / self.lam).view(-1, 1, 1, 1) * self.delta_r
        return torch.clamp(img, 0.0, 1.0)


class WhitenedLatentSpace(AttackSpace):
    """RAG latent space in gain-whitened coordinates w:  z = z_s + s * w.

    s_j = sqrt(mean(d) / d_j) with d = diag(J^T J) (decoder gain per latent
    coordinate). Euclidean distance in w approximates the pixel distance
    (Jacobi / diagonal pull-back metric), so geometric attacks that minimise
    ||w|| no longer drift towards high-gain (pixel-expensive) latent directions.
    """

    name = "latent_w"
    is_latent = True

    def __init__(self, z_s: torch.Tensor, x0: torch.Tensor, decoder: nn.Module, gain: torch.Tensor,
                 floor: float = 1e-3):
        super().__init__(origin=torch.zeros_like(z_s), x0=x0)
        self.z_s = z_s.detach()
        self.decoder = decoder
        d = gain.detach().clamp_min(0.0)
        d = d / d.mean().clamp_min(1e-20)
        self.scale = (1.0 / d.clamp_min(floor)).sqrt()  # <= 1/sqrt(floor)
        with torch.no_grad():
            self.x_rec0 = decoder(self.z_s).detach()

    def to_latent(self, w: torch.Tensor) -> torch.Tensor:
        return self.z_s + self.scale * w

    @torch.no_grad()
    def to_image(self, w: torch.Tensor) -> torch.Tensor:
        return torch.clamp(self.x0 + self.decoder(self.to_latent(w)) - self.x_rec0, 0.0, 1.0)


class QueryBudgetExhausted(RuntimeError):
    pass


class HardLabelOracle:
    """Counted top-1 oracle. Only +1 (adversarial) / -1 decisions leave this class."""

    def __init__(
        self,
        classifier: nn.Module,
        space: AttackSpace,
        source_label: int,
        target_label: Optional[int],
        max_queries: int,
        batch_size: int = 64,
    ):
        self.classifier = classifier
        self.space = space
        self.source_label = int(source_label)
        self.target_label = None if target_label is None else int(target_label)
        self.max_queries = int(max_queries)
        self.batch_size = max(1, int(batch_size))
        self.queries = 0
        self.by_tag: Counter = Counter()
        self.device = space.origin.device

    @property
    def remaining(self) -> int:
        return max(0, self.max_queries - self.queries)

    def _adv_from_labels(self, labels: torch.Tensor) -> torch.Tensor:
        if self.target_label is None:
            return labels != self.source_label
        return labels == self.target_label

    @torch.no_grad()
    def _labels(self, z: torch.Tensor) -> torch.Tensor:
        out = []
        i = 0
        while i < z.shape[0]:
            b = min(self.batch_size, z.shape[0] - i)
            try:
                logits = self.classifier(self.space.to_image(z[i : i + b]))
            except RuntimeError as err:
                if is_oom(err) and self.batch_size > 1:
                    self.batch_size = max(1, self.batch_size // 2)
                    empty_cache(self.device)
                    print(f"      [oracle] OOM -> query batch size reduced to {self.batch_size}")
                    continue
                raise
            out.append(logits.argmax(1))
            i += b
        return torch.cat(out)

    def is_adv(self, z: torch.Tensor, tag: str = "update") -> bool:
        """One counted query on a single point z of shape (1, *shape)."""
        if self.remaining <= 0:
            raise QueryBudgetExhausted
        label = self._labels(z)
        self.queries += 1
        self.by_tag[tag] += 1
        return bool(self._adv_from_labels(label)[0].item())

    def signs(self, z_batch: torch.Tensor, tag: str = "normal") -> torch.Tensor:
        """Counted batch query; returns float +1/-1 of shape (B,). Caller caps B."""
        q = int(z_batch.shape[0])
        if q <= 0:
            return torch.empty(0, device=self.device)
        if q > self.remaining:
            raise QueryBudgetExhausted
        labels = self._labels(z_batch)
        self.queries += q
        self.by_tag[tag] += q
        adv = self._adv_from_labels(labels)
        return adv.float() * 2.0 - 1.0

    @torch.no_grad()
    def label_uncounted(self, image: torch.Tensor) -> int:
        """Evaluation-only label of an IMAGE (not counted, never used by attacks)."""
        return int(self.classifier(image).argmax(1)[0].item())

    def is_adv_label(self, label: int) -> bool:
        if self.target_label is None:
            return int(label) != self.source_label
        return int(label) == self.target_label
