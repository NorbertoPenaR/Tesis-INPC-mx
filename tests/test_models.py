import numpy as np
import pytest
import torch

from inpc_forecasting.factory import create_model
from inpc_forecasting.models.transformer import TransformerForecast


@pytest.mark.parametrize("name", ["rnn", "lstm", "deepar", "transformer"])
def test_models_train_and_predict_finite_values(name):
    rng = np.random.default_rng(119)
    history = rng.normal(size=(10, 12, 3)).astype("float32")
    future = rng.normal(size=(10, 3, 2)).astype("float32")
    target = rng.normal(size=(10, 3)).astype("float32")
    architecture = {"hidden_size": 8, "num_layers": 1, "dropout": 0.0}
    if name == "transformer":
        architecture = {"d_model": 8, "num_layers": 1, "nhead": 2, "dim_feedforward": 16, "dropout": 0.0}
    config = {
        "architecture": architecture,
        "training": {"epochs": 1, "batch_size": 5, "patience": 1, "device": "cpu", "seed": 119},
    }
    model = create_model(name, input_size=3, exog_size=2, horizon=3, config=config)
    model.fit(history, future, target)
    prediction = model.predict(history[-1], future[-1])
    assert prediction.shape == (3,)
    assert np.isfinite(prediction).all()
    assert all(parameter.grad is None or torch.isfinite(parameter.grad).all() for parameter in model.parameters())


def test_deepar_scale_is_positive():
    model = create_model(
        "deepar", 2, 1, 3,
        {"architecture": {"hidden_size": 8}, "training": {"epochs": 1, "device": "cpu"}},
    )
    _, scale = model(torch.zeros(2, 8, 2), torch.zeros(2, 3, 1))
    assert torch.all(scale > 0)


def test_transformer_mask_blocks_future_positions():
    mask = TransformerForecast.causal_mask(4)
    assert torch.isneginf(mask[0, 1:]).all()
    assert torch.equal(torch.diag(mask), torch.zeros(4))


@pytest.mark.parametrize("name", ["rnn", "lstm", "deepar", "transformer"])
def test_model_initialization_is_reproducible(name):
    architecture = {"hidden_size": 8, "num_layers": 1, "dropout": 0.0}
    if name == "transformer":
        architecture = {"d_model": 8, "num_layers": 1, "nhead": 2, "dim_feedforward": 16, "dropout": 0.0}
    config = {"architecture": architecture, "training": {"seed": 119, "device": "cpu"}}

    torch.manual_seed(999)
    first = create_model(name, input_size=3, exog_size=2, horizon=3, config=config)
    torch.manual_seed(1234)
    second = create_model(name, input_size=3, exog_size=2, horizon=3, config=config)

    for first_parameter, second_parameter in zip(first.parameters(), second.parameters()):
        torch.testing.assert_close(first_parameter, second_parameter)
