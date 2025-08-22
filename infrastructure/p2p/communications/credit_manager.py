# SPDX-License-Identifier: Apache-2.0
"""CreditManager → mints & spends compute tokens on Bittensor."""

try:
    from bittensor_wallet import Network, Wallet  # pip install bittensor-wallet
except ImportError as e:  # pragma: no cover - runtime dependency
    Network = Wallet = None
    _BITTENSOR_IMPORT_ERROR: ImportError | None = e
else:
    _BITTENSOR_IMPORT_ERROR = None


class CreditManager:
    def __init__(self, mnemonic: str) -> None:
        """Create a credit manager bound to a wallet mnemonic."""
        if _BITTENSOR_IMPORT_ERROR is not None:
            msg = (
                "bittensor-wallet is required to use CreditManager. " "Install it with 'pip install bittensor-wallet'."
            )
            raise ImportError(msg) from _BITTENSOR_IMPORT_ERROR
        self.wallet = Wallet.from_mnemonic(mnemonic)
        self.network = Network(self.wallet)

    def mint(self, _task_id: str, macs: int) -> str:
        """1 credit = 1e12 MACs.

        Returns tx hash.
        """
        tokens = macs / 1e12
        return self.network.reward_miner(self.wallet, tokens)

    def spend(self, to_ss58: str, amount: float) -> str:
        """Transfer tokens; returns tx hash."""
        return self.network.transfer(self.wallet, to_ss58, amount)

    def balance(self) -> float:
        return self.network.get_balance(self.wallet)
