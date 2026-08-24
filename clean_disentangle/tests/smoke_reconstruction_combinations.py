#!/usr/bin/env python3
"""Deterministic six-way composition smoke test, including num_t > 1 mapping."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from torch import nn

from clean_disentangle.engine import train_reconstruction_step
from clean_disentangle.modeling import (
    FULL_DIRECT_SPEC,
    FULL_PROTOTYPE_SPEC,
    MISSING_DIRECT_SPEC,
    MISSING_PROTOTYPE_SPEC,
    ComponentMode,
    CompositionMode,
    MissingFillMode,
    OutputBaseMode,
    ReconstructionModel,
    ReconstructionScope,
    ReconstructionSpec,
    StableCore,
)
from clean_disentangle.prototype import (
    PrototypeProvider,
    channel_positions_to_token_positions,
    select_positions,
)


class DeterministicPatchEmbed(nn.Module):
    """Small frozen tokenizer that preserves channel-major/time-major ordering."""

    def __init__(self, embed_dim: int) -> None:
        super().__init__()
        self.embed_dim = int(embed_dim)

    def forward(self, eeg: torch.Tensor) -> torch.Tensor:
        if eeg.ndim != 4:
            raise ValueError(f"expected [B,C,num_t,patch], got {tuple(eeg.shape)}")
        mean = eeg.mean(dim=-1)
        first = eeg[..., 0]
        features = [
            mean * float(index + 1) + first * float(index + 2) / 10.0
            for index in range(self.embed_dim)
        ]
        return torch.stack(features, dim=-1).flatten(1, 2)


def make_fixture(
    *,
    with_prototype: bool = True,
) -> tuple[ReconstructionModel, dict[str, torch.Tensor]]:
    torch.manual_seed(17)
    batch_size = 3
    full_channels = 4
    observed_positions = (0, 2)
    missing_positions = (1, 3)
    num_t = 2
    patch_size = 3
    embed_dim = 4

    x_full = torch.arange(
        batch_size * full_channels * num_t * patch_size,
        dtype=torch.float32,
    ).reshape(batch_size, full_channels, num_t * patch_size)
    x_full = x_full / 50.0 - 1.0
    x_obs = x_full[:, observed_positions, :].clone()
    prototype_bank = torch.arange(
        full_channels * num_t * embed_dim,
        dtype=torch.float32,
    ).reshape(full_channels, num_t, embed_dim)
    prototype_bank = prototype_bank / 20.0 + 0.25

    provider = (
        PrototypeProvider(
            prototype_bank,
            channel_names=("A", "B", "C", "D"),
        )
        if with_prototype
        else None
    )
    model = ReconstructionModel(
        patch_embed=DeterministicPatchEmbed(embed_dim),
        stable_core=StableCore(
            embed_dim=embed_dim,
            num_layers=1,
            num_heads=2,
            dropout=0.0,
        ),
        prototype_provider=provider,
        full_num_channels=full_channels,
        observed_channel_positions=observed_positions,
        missing_channel_positions=missing_positions,
        patch_size=patch_size,
        num_t=num_t,
    )
    return model, {"x_full": x_full, "x_obs": x_obs}


def assert_close(actual: torch.Tensor, expected: torch.Tensor, label: str) -> float:
    if actual.shape != expected.shape:
        raise AssertionError(
            f"{label} shape mismatch: {tuple(actual.shape)} vs {tuple(expected.shape)}"
        )
    max_abs_diff = float((actual - expected).abs().max().item())
    if max_abs_diff > 1e-7:
        raise AssertionError(f"{label} max_abs_diff={max_abs_diff:.9g}")
    return max_abs_diff


def main() -> None:
    torch.set_num_threads(1)
    model, batch = make_fixture()
    model.eval()

    invalid_specs = (
        dict(
            scope=ReconstructionScope.FULL,
            missing_fill=MissingFillMode.ZERO,
            output_base=OutputBaseMode.NONE,
        ),
        dict(
            scope=ReconstructionScope.MISSING,
            missing_fill=MissingFillMode.NOT_APPLICABLE,
            output_base=OutputBaseMode.NONE,
        ),
    )
    for invalid in invalid_specs:
        try:
            ReconstructionSpec(**invalid)
        except ValueError:
            pass
        else:
            raise AssertionError(f"invalid spec was accepted: {invalid}")

    missing_input_only = ReconstructionSpec(
        scope=ReconstructionScope.MISSING,
        missing_fill=MissingFillMode.PROTOTYPE,
        output_base=OutputBaseMode.NONE,
        component_mode=ComponentMode.IDENTITY,
        composition_mode=CompositionMode.SUM,
    )
    missing_output_only = ReconstructionSpec(
        scope=ReconstructionScope.MISSING,
        missing_fill=MissingFillMode.ZERO,
        output_base=OutputBaseMode.PROTOTYPE,
        component_mode=ComponentMode.IDENTITY,
        composition_mode=CompositionMode.SUM,
    )
    cases = (
        ("full_none", FULL_DIRECT_SPEC),
        ("full_prototype", FULL_PROTOTYPE_SPEC),
        ("missing_prototype_prototype", MISSING_PROTOTYPE_SPEC),
        ("missing_zero_none", MISSING_DIRECT_SPEC),
        ("missing_prototype_none", missing_input_only),
        ("missing_zero_prototype", missing_output_only),
    )

    expected_observed_tokens = torch.tensor([0, 1, 4, 5])
    expected_missing_tokens = torch.tensor([2, 3, 6, 7])
    if not torch.equal(
        channel_positions_to_token_positions((0, 2), num_t=2),
        expected_observed_tokens,
    ):
        raise AssertionError("observed channel-to-token mapping is incorrect")
    if not torch.equal(
        channel_positions_to_token_positions((1, 3), num_t=2),
        expected_missing_tokens,
    ):
        raise AssertionError("missing channel-to-token mapping is incorrect")

    full_tokens = model.patch_tokens(batch["x_full"], expected_channels=4)
    observed_tokens = model.patch_tokens(batch["x_obs"], expected_channels=2)
    provider = model.prototype_provider
    if provider is None:
        raise AssertionError("prototype fixture unexpectedly lacks a provider")
    full_prototype = provider.get_full(
        batch_size=batch["x_full"].shape[0],
        num_t=2,
        device=full_tokens.device,
        dtype=full_tokens.dtype,
    )
    missing_prototype = select_positions(full_prototype, expected_missing_tokens)
    provider_missing = provider.get_missing(
        batch_size=batch["x_full"].shape[0],
        missing_channel_positions=(1, 3),
        num_t=2,
        device=full_tokens.device,
        dtype=full_tokens.dtype,
    )
    assert_close(provider_missing, missing_prototype, "provider full/missing selection")

    outputs: dict[str, dict] = {}
    rows: list[dict[str, object]] = []
    with torch.no_grad():
        for name, spec in cases:
            output = model.forward_reconstruction(batch, spec)
            outputs[name] = output
            if output["pred"].shape != output["target"].shape:
                raise AssertionError(f"{name}: prediction/target shape mismatch")
            if not torch.isfinite(output["pred"]).all() or not torch.isfinite(
                output["target"]
            ).all():
                raise AssertionError(f"{name}: non-finite prediction or target")

            expected_positions = (
                torch.arange(8)
                if spec.scope is ReconstructionScope.FULL
                else expected_missing_tokens
            )
            if not torch.equal(output["target_positions"].cpu(), expected_positions):
                raise AssertionError(f"{name}: target token mapping differs")
            expected_target = select_positions(full_tokens, expected_positions)
            assert_close(output["target"], expected_target, f"{name} target")

            if spec.scope is ReconstructionScope.FULL:
                assert_close(output["input_tokens"], full_tokens, f"{name} full input")
                expected_prototype = full_prototype
            else:
                actual_observed = select_positions(
                    output["input_tokens"],
                    expected_observed_tokens,
                )
                assert_close(actual_observed, observed_tokens, f"{name} observed placement")
                actual_missing = select_positions(
                    output["input_tokens"],
                    expected_missing_tokens,
                )
                expected_fill = (
                    missing_prototype
                    if spec.missing_fill is MissingFillMode.PROTOTYPE
                    else torch.zeros_like(missing_prototype)
                )
                assert_close(actual_missing, expected_fill, f"{name} missing fill")
                expected_prototype = missing_prototype
            if spec.requires_prototype:
                assert_close(
                    output["prototype_selected"],
                    expected_prototype,
                    f"{name} prototype selection",
                )
            elif output["prototype_selected"] is not None:
                raise AssertionError(f"{name}: unused prototype was accessed")

            manual_d_sub = select_positions(output["sub_tokens"], expected_positions)
            manual_d_task = select_positions(output["task_tokens"], expected_positions)
            assert_close(output["d_sub"], manual_d_sub, f"{name} d_sub")
            assert_close(output["d_task"], manual_d_task, f"{name} d_task")
            manual_base = (
                torch.zeros_like(output["target"])
                if spec.output_base is OutputBaseMode.NONE
                else expected_prototype
            )
            manual_pred = manual_base + manual_d_sub + manual_d_task
            diff = assert_close(output["pred"], manual_pred, f"{name} manual prediction")
            rows.append(
                {
                    "case": name,
                    "pred_shape": list(output["pred"].shape),
                    "manual_max_abs_diff": diff,
                    "finite": True,
                }
            )

    # Same scope/fill must yield identical representations when only output base changes.
    assert_close(
        outputs["full_none"]["input_tokens"],
        outputs["full_prototype"]["input_tokens"],
        "full base-policy input invariance",
    )
    assert_close(
        outputs["full_none"]["sub_tokens"],
        outputs["full_prototype"]["sub_tokens"],
        "full base-policy StableCore invariance",
    )
    assert_close(
        outputs["missing_prototype_prototype"]["sub_tokens"],
        outputs["missing_prototype_none"]["sub_tokens"],
        "prototype-fill output-base independence",
    )
    assert_close(
        outputs["missing_zero_none"]["sub_tokens"],
        outputs["missing_zero_prototype"]["sub_tokens"],
        "zero-fill output-base independence",
    )

    # Named APIs are thin semantic aliases for the same spec-driven pipeline.
    with torch.no_grad():
        named = {
            "full_none": model.forward_full_direct(batch),
            "full_prototype": model.forward_full_prototype(batch),
            "missing_zero_none": model.forward_missing_direct(batch),
            "missing_prototype_prototype": model.forward_missing_prototype(batch),
        }
    for name, output in named.items():
        assert_close(output["pred"], outputs[name]["pred"], f"{name} named wrapper")

    # A model with no provider must run every spec that does not require one.
    no_provider_model, no_provider_batch = make_fixture(with_prototype=False)
    no_provider_model.eval()
    no_provider_rows = []
    with torch.no_grad():
        for name, spec in (
            ("full_none", FULL_DIRECT_SPEC),
            ("missing_zero_none", MISSING_DIRECT_SPEC),
        ):
            output = no_provider_model.forward_reconstruction(no_provider_batch, spec)
            if output["prototype_selected"] is not None:
                raise AssertionError(f"{name}: no-provider output contains a prototype")
            if output["pred"].shape != output["target"].shape:
                raise AssertionError(f"{name}: no-provider prediction/target mismatch")
            if not torch.isfinite(output["pred"]).all():
                raise AssertionError(f"{name}: no-provider prediction is non-finite")
            no_provider_rows.append(name)

        for name, spec in (
            ("full_prototype", FULL_PROTOTYPE_SPEC),
            ("missing_prototype", MISSING_PROTOTYPE_SPEC),
        ):
            try:
                no_provider_model.forward_reconstruction(no_provider_batch, spec)
            except RuntimeError as error:
                if "no PrototypeProvider" not in str(error):
                    raise AssertionError(
                        f"{name}: unexpected missing-provider error: {error}"
                    ) from error
            else:
                raise AssertionError(f"{name}: prototype spec accepted no provider")

    # One real generic engine step: it receives only model, batch, spec, and optimizer.
    first_parameter = next(model.stable_core.parameters())
    before = first_parameter.detach().clone()
    optimizer = torch.optim.AdamW(model.stable_core.parameters(), lr=1e-3)
    _, engine_metrics = train_reconstruction_step(
        model,
        batch,
        FULL_DIRECT_SPEC,
        optimizer,
    )
    if not torch.isfinite(torch.tensor(engine_metrics["loss"])):
        raise AssertionError("generic engine returned non-finite loss")
    if torch.equal(before, first_parameter.detach()):
        raise AssertionError("generic engine step did not update StableCore")
    if model.patch_embed.training:
        raise AssertionError("frozen patch embedding left eval mode during model.train()")

    # Static boundary guard: engine and loss must not contain composition internals.
    package_root = Path(__file__).resolve().parents[1]
    engine_source = (package_root / "engine.py").read_text(encoding="utf-8")
    for forbidden in (
        "OutputBaseMode",
        "MissingFillMode",
        "compose_prediction",
        "select_positions",
        "prototype_selected",
        'output["d_sub"]',
        'output["d_task"]',
    ):
        if forbidden in engine_source:
            raise AssertionError(f"engine contains forbidden policy detail: {forbidden}")
    loss_source = (package_root / "losses.py").read_text(encoding="utf-8").lower()
    if "prototype" in loss_source:
        raise AssertionError("losses.py contains prototype knowledge")

    print(
        json.dumps(
            {
                "cases": rows,
                "no_provider_cases": no_provider_rows,
                "engine": engine_metrics,
            },
            indent=2,
        )
    )
    print("SMOKE PASS: six orthogonal reconstruction combinations")


if __name__ == "__main__":
    main()
