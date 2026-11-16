import pytest
import torch

from arb_stem_mss.inference import overlap_add
from arb_stem_mss.losses import SeparationLoss
from arb_stem_mss.metrics import compute_metrics, aggregate_metrics


def loss_function():
    return SeparationLoss({"wave_type": "l1", "wave_weight": 1.0,
                           "mrstft_weight": 1.0, "fft_sizes": [32, 64],
                           "hop_sizes": [8, 16], "win_lengths": [32, 64],
                           "absence_weight": 0.1})


def test_loss_is_finite_on_short_silent_audio_and_masks_padding():
    criterion = loss_function()
    predicted = torch.zeros(2, 1, 9, requires_grad=True)
    target = torch.zeros_like(predicted)
    result = criterion(predicted, target, num_samples=torch.tensor([9, 3]),
                       present=torch.tensor([True, False]))
    assert torch.isfinite(result["loss"]) and result["loss"] < 1e-5
    result["loss"].backward()
    assert torch.isfinite(predicted.grad).all()
    a = torch.ones(1, 1, 96) * 0.1
    b = a.clone()
    b[..., 41:] = 1000  # Invalid right-padding must not influence any loss.
    target = torch.zeros_like(a)
    lengths = torch.tensor([41])
    original = criterion(a, target, num_samples=lengths)["loss"]
    padded = criterion(b, target, num_samples=lengths)["loss"]
    assert torch.allclose(original, padded, atol=1e-6)


@pytest.mark.parametrize("frames", [1, 127, 128, 129, 511])
def test_overlap_add_identity_keeps_both_boundaries(frames):
    class Identity(torch.nn.Module):
        def forward(self, mixture, query):
            return mixture
    x = torch.linspace(-0.4, 0.6, frames)[None]
    output = overlap_add(Identity(), x, torch.ones(8), segment_samples=128,
                         device="cpu", overlap=0.5)
    assert output.shape == x.shape
    assert torch.allclose(output, x, atol=1e-6)
    assert torch.allclose(output[..., [0, -1]], x[..., [0, -1]], atol=1e-6)


def test_absent_and_silent_targets_excluded_from_ratio_metrics():
    absent = compute_metrics(torch.ones(1, 32) * 0.01, torch.zeros(1, 32),
                             mixture=torch.ones(1, 32) * 0.2, present=False)
    assert absent["si_sdr_db"] is None and absent["sdr_db"] is None
    assert absent["output_rms"] > 0
    silent = compute_metrics(torch.zeros(1, 32), torch.zeros(1, 32), present=True)
    assert silent["si_sdr_db"] is None


def test_metric_aggregation_weights_songs_instead_of_query_count():
    rows = []
    for song, count, score in (("a", 3, 10.0), ("b", 1, 0.0)):
        for index in range(count):
            rows.append({"song_id": song, "si_sdr_db": score, "sdr_db": score,
                         "target_present": True, "target_silent": False,
                         "mse": 0.1, "l1": 0.1, "output_rms": 0.1,
                         "target_rms": 0.1, "relative_output_energy_db": 0.0})
    result = aggregate_metrics(rows)
    assert result["num_songs"] == 2 and result["num_queries"] == 4
    assert result["macro_by_song"]["si_sdr_db"] == pytest.approx(5.0)
