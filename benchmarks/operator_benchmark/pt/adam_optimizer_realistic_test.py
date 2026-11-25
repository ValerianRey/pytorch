import operator_benchmark as op_bench

import torch
import torch.optim as optim


"""Microbenchmarks for Adam (for-loop, foreach, fused) on parameter sets of real models."""


# (count, shape) groups of the parameters updated in one step.
adam_settings = {
    # Hidden matrices only, the same sets as the Muon benchmark.
    "gpt2_small_124m": [(48, (768, 768)), (12, (3072, 768)), (12, (768, 3072))],
    "llama3_2_1b": [
        (32, (2048, 2048)),
        (32, (512, 2048)),
        (32, (8192, 2048)),
        (16, (2048, 8192)),
    ],
    "moonlight_moe_layer": [
        (128, (1408, 2048)),
        (64, (2048, 1408)),
        (2, (2816, 2048)),
        (1, (2048, 2816)),
    ],
    # Llama-3.2-1B including what Adam also updates in practice: the (tied)
    # embedding and the 33 RMSNorm weight vectors.
    "llama3_2_1b_full": [
        (32, (2048, 2048)),
        (32, (512, 2048)),
        (32, (8192, 2048)),
        (16, (2048, 8192)),
        (1, (128256, 2048)),
        (33, (2048,)),
    ],
    "single_2048x2048": [(1, (2048, 2048))],
}

adam_impls = op_bench.op_list(
    attr_names=["op_name", "op_func"],
    attrs=[
        ["adam_forloop", "forloop"],
        ["adam_foreach", "foreach"],
        ["adam_fused", "fused"],
    ],
)

adam_configs = op_bench.config_list(
    attr_names=["setting"],
    attrs=[[setting] for setting in adam_settings],
    cross_product_configs={
        "device": ["cuda"],
    },
    tags=["long"],
)

_impl_kwargs = {
    "forloop": {"foreach": False, "fused": False},
    "foreach": {"foreach": True, "fused": False},
    "fused": {"foreach": False, "fused": True},
}


class AdamOptimizerRealisticBenchmark(op_bench.TorchBenchmarkBase):
    def init(self, op_func, device, setting):
        self.params = [
            torch.nn.Parameter(torch.randn(*shape, device=device, dtype=torch.float32))
            for count, shape in adam_settings[setting]
            for _ in range(count)
        ]
        for p in self.params:
            p.grad = torch.randn_like(p)

        self.optimizer = optim.Adam(self.params, lr=0.001, **_impl_kwargs[op_func])
        # Initialize optimizer state before benchmarking steady-state step time.
        self.optimizer.step()
        for p in self.params:
            p.grad = torch.randn_like(p)

        self.inputs = {"dummy": self.params[0]}

    def forward(self, dummy):
        self.optimizer.step()
        return dummy

    def get_memory_traffic_bytes(self):
        return None


op_bench.generate_pt_tests_from_op_list(
    adam_impls, adam_configs, AdamOptimizerRealisticBenchmark
)


if __name__ == "__main__":
    op_bench.benchmark_runner.main()
