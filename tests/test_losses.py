"""Test #24: KD direction KL(teacher || student)."""

import torch

from mnist1d_cl.losses import kd_loss


def test_kd_zero_when_equal():
    z = torch.randn(8, 10)
    assert torch.allclose(kd_loss(z, z, temperature=2.0),
                          torch.tensor(0.0), atol=1e-6)


def test_kd_positive_and_asymmetric():
    z1 = torch.randn(8, 10)
    z2 = torch.randn(8, 10)
    a = kd_loss(z1, z2, temperature=2.0)   # student=z1, teacher=z2
    b = kd_loss(z2, z1, temperature=2.0)   # student=z2, teacher=z1
    assert a.item() > 0
    # direction matters: KL(teacher||student) is asymmetric
    assert not torch.allclose(a, b, atol=1e-4)


def test_kd_class_restriction():
    z1 = torch.randn(8, 10)
    z2 = torch.randn(8, 10)
    val = kd_loss(z1, z2, temperature=2.0, classes=[0, 1, 2, 3, 4])
    assert val.dim() == 0 and torch.isfinite(val)


def test_kd_matches_manual_direction():
    import torch.nn.functional as F
    student = torch.randn(4, 6)
    teacher = torch.randn(4, 6)
    T = 3.0
    manual = F.kl_div(F.log_softmax(student / T, dim=1),
                      F.softmax(teacher / T, dim=1),
                      reduction="batchmean") * (T * T)
    assert torch.allclose(kd_loss(student, teacher, temperature=T), manual, atol=1e-6)
