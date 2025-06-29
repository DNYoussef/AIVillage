# SPDX-License-Identifier: Apache-2.0
"""
CreditManager → mints & spends compute tokens on Bittensor.
"""

from bittensor_wallet import Wallet, Network  # pip install bittensor-wallet


class CreditManager:
    def __init__(self, mnemonic: str):
        self.wallet = Wallet.from_mnemonic(mnemonic)
        self.network = Network(self.wallet)

    def mint(self, task_id: str, macs: int) -> str:
        """
        1 credit = 1e12 MACs
        Returns tx hash.
        """
        tokens = macs / 1e12
        return self.network.reward_miner(self.wallet, tokens)

    def spend(self, to_ss58: str, amount: float) -> str:
        """Transfer tokens; returns tx hash."""
        return self.network.transfer(self.wallet, to_ss58, amount)

    def balance(self) -> float:
        return self.network.get_balance(self.wallet)

