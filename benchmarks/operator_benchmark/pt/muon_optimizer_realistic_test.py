import operator_benchmark as op_bench

import torch
import torch.optim as optim


"""Microbenchmarks for the Muon optimizer on parameter sets taken from real models."""


# (count, rows, cols) groups of the 2D hidden weights that Muon updates in one step.
# Embeddings, the LM head, norms and biases are handled by AdamW in practice and are
# not included.
muon_settings = {
    # GPT-2 small: 12 layers, d_model 768, 4x MLP; q, k, v, o as separate matrices.
    "gpt2_small_124m": [(48, 768, 768), (12, 3072, 768), (12, 768, 3072)],
    # Llama-3.2-1B: 16 layers, d_model 2048, 8 KV heads of dim 64, MLP 8192.
    "llama3_2_1b": [
        (32, 2048, 2048),
        (32, 512, 2048),
        (32, 8192, 2048),
        (16, 2048, 8192),
    ],
    # Moonlight-16B-A3B, experts of one MoE layer: 64 routed experts (d_model 2048,
    # expert dim 1408) plus the shared expert (2 x 1408).
    "moonlight_moe_layer": [
        (128, 1408, 2048),
        (64, 2048, 1408),
        (2, 2816, 2048),
        (1, 2048, 2816),
    ],
}

muon_impls = op_bench.op_list(
    attr_names=["op_name", "op_func"],
    attrs=[
        ["muon_forloop", False],
        ["muon_foreach", True],
    ],
)

muon_configs = op_bench.config_list(
    attr_names=["setting"],
    attrs=[[setting] for setting in muon_settings],
    cross_product_configs={
        "device": ["cuda"],
    },
    tags=["long"],
)


class MuonOptimizerRealisticBenchmark(op_bench.TorchBenchmarkBase):
    def init(self, op_func, device, setting):
        self.params = [
            torch.nn.Parameter(
                torch.randn(rows, cols, device=device, dtype=torch.float32)
            )
            for count, rows, cols in muon_settings[setting]
            for _ in range(count)
        ]
        for p in self.params:
            p.grad = torch.randn_like(p)

        self.optimizer = optim.Muon(self.params, lr=0.001, foreach=op_func)
        # Initialize momentum buffers before benchmarking steady-state step time.
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
    muon_impls, muon_configs, MuonOptimizerRealisticBenchmark
)


if __name__ == "__main__":
    op_bench.benchmark_runner.main()
