from __future__ import annotations

from pathlib import Path

import pytest
import torch
import yaml

from solar_ir.finetuning import apply_finetuning_policy
from solar_ir.models.dnanet import create_model
from solar_ir.optimization import build_optimizer


def test_frozen_dnanet_parameters_and_bn_stay_fixed_after_training_steps():
    torch.set_num_threads(2)
    model = create_model({"input_channels": 1, "channel_size": "one", "backbone": "resnet_10"})
    policy = {
        "mode": "freeze_modules",
        "modules": ["conv0_0", "conv1_0", "conv2_0"],
        "freeze_running_stats": True,
    }
    metadata = apply_finetuning_policy(model, policy)
    before = {name: tensor.clone() for name, tensor in model.state_dict().items()}
    optimizer = build_optimizer(model, {"optimizer": "Adagrad", "learning_rate": 0.005})
    optimized = {id(p) for group in optimizer.param_groups for p in group["params"]}
    assert all((id(p) in optimized) == p.requires_grad for p in model.parameters())
    assert metadata["frozen_parameters"] > 0
    for _ in range(2):
        model.train()
        optimizer.zero_grad()
        outputs = model(torch.randn(2, 1, 32, 32))
        sum(output.square().mean() for output in outputs).backward()
        assert all(
            p.grad is None
            for name in policy["modules"]
            for p in model.get_submodule(name).parameters()
        )
        optimizer.step()
    for name, tensor in model.state_dict().items():
        if any(name.startswith(module + ".") for module in policy["modules"]):
            assert torch.equal(tensor, before[name]), name
    assert not torch.equal(model.final4.weight, before["final4.weight"])
    assert not torch.equal(model.conv3_0[0].bn1.running_mean, before["conv3_0.0.bn1.running_mean"])


def test_full_finetuning_has_no_hooks_or_frozen_parameters():
    model = create_model({"input_channels": 1, "channel_size": "one"})
    metadata = apply_finetuning_policy(model, {"mode": "full"})
    assert metadata["frozen_parameters"] == 0
    assert all(p.requires_grad for p in model.parameters())
    assert not any(module._forward_pre_hooks for module in model.modules())


@pytest.mark.parametrize(
    "policy",
    [
        {"mode": "freeze_modules", "modules": ["missing"]},
        {"mode": "full", "modules": ["conv0_0"]},
        {"mode": "freeze_modules", "modules": []},
    ],
)
def test_invalid_finetuning_policy_fails(policy):
    with pytest.raises(ValueError):
        apply_finetuning_policy(create_model({"channel_size": "one"}), policy)


def test_paired_finetuning_configs_change_only_policy_and_run_identity():
    root = Path(__file__).resolve().parents[1] / "configs/pvhsd2025"
    base = yaml.safe_load((root / "finetune_solar_dnanet_1ch_cudnn.yaml").read_text())
    for variant in ["full", "freeze_early"]:
        config = yaml.safe_load(
            (root / f"finetune_solar_dnanet_1ch_group_{variant}.yaml").read_text()
        )
        config.pop("name")
        config["training"].pop("finetuning")
        expected = dict(base)
        expected.pop("name")
        assert config == expected
