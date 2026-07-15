"""Knowledge distillation loss with the standard KL(teacher || student) direction.

The teacher is the target distribution. In PyTorch, ``F.kl_div`` expects the
*input* as log-probabilities and the *target* as probabilities, and computes
``sum target * (log target - input)`` i.e. ``KL(target || input)``. With
input = log_softmax(student) and target = softmax(teacher), this is exactly
``KL(teacher || student)``.
"""

from __future__ import annotations

from typing import Optional, Sequence

import torch
import torch.nn.functional as F


def kd_loss(student_logits: torch.Tensor, teacher_logits: torch.Tensor,
            temperature: float = 2.0,
            classes: Optional[Sequence[int]] = None,
            reduction: str = "batchmean") -> torch.Tensor:
    """KL(teacher || student) distillation loss, scaled by T**2.

    If ``classes`` is given, the distributions are restricted (and renormalized)
    to those class logits, i.e. distillation over the "old" classes only.
    """
    if temperature <= 0:
        raise ValueError("temperature must be > 0")
    if classes is not None:
        idx = torch.as_tensor(list(classes), device=student_logits.device,
                              dtype=torch.long)
        student_logits = student_logits.index_select(1, idx)
        teacher_logits = teacher_logits.index_select(1, idx)

    T = float(temperature)
    log_p_student = F.log_softmax(student_logits / T, dim=1)   # input  = log-prob student
    p_teacher = F.softmax(teacher_logits / T, dim=1)           # target = prob teacher
    return F.kl_div(log_p_student, p_teacher, reduction=reduction) * (T * T)
