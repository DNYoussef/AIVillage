# SPDX-License-Identifier: Apache-2.0
"""
ShardPlanner: ILP for layer→node assignment minimizing makespan.
"""

import pulp


class ShardPlanner:
    def __init__(self,
                 layer_flops: dict[int, float],
                 node_speeds: dict[str, float]):
        """
        layer_flops: {layer_idx: flops_required}
        node_speeds: {peer_id: flops_per_sec}
        """
        self.layer_flops = layer_flops
        self.node_speeds = node_speeds

    def plan(self) -> dict[int, str]:
        prob = pulp.LpProblem("ShardPlanning", pulp.LpMinimize)
        layers = list(self.layer_flops)
        nodes = list(self.node_speeds)

        # vars x[l,n] ∈ {0,1}
        x = {(l,n): pulp.LpVariable(f"x_{l}_{n}", cat="Binary")
             for l in layers for n in nodes}

        T = pulp.LpVariable("Makespan", lowBound=0)

        # objective
        prob += T

        # each layer exactly one node
        for l in layers:
            prob += pulp.lpSum(x[(l,n)] for n in nodes) == 1

        # per-node load constraint
        for n in nodes:
            prob += (
                pulp.lpSum(self.layer_flops[l] / self.node_speeds[n] * x[(l,n)]
                           for l in layers)
                <= T
            )

        prob.solve(pulp.PULP_CBC_CMD(msg=False))

        assignment = {}
        for l in layers:
            for n in nodes:
                if pulp.value(x[(l,n)]) > 0.5:
                    assignment[l] = n
        return assignment

