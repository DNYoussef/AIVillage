import pytest
from contracts.VILLAGEToken import VILLAGEToken
from web3 import Web3


@pytest.mark.asyncio
async def test_token_economy_integration():
    # Create a web3 instance
    w3 = Web3(Web3.EthereumTesterProvider())

    # Deploy the token contract
    token_contract = w3.eth.contract(
        abi=VILLAGEToken.abi, bytecode=VILLAGEToken.bytecode
    )
    tx_hash = token_contract.constructor().transact(
        {"from": w3.eth.accounts[0], "gas": 1000000}
    )
    tx_receipt = w3.eth.waitForTransactionReceipt(tx_hash)
    token_address = tx_receipt.contractAddress
    token = w3.eth.contract(address=token_address, abi=VILLAGEToken.abi)

    # Mint some tokens
    token.functions.mintForEducation(w3.eth.accounts[1], 100).transact(
        {"from": w3.eth.accounts[0]}
    )

    # Check the balance
    balance = token.functions.balanceOf(w3.eth.accounts[1]).call()
    assert balance == 100

    # Stake some tokens
    token.functions.stakeForLearning(50).transact({"from": w3.eth.accounts[1]})

    # Check the staking balance
    staking_balance = token.functions.stakingBalance(w3.eth.accounts[1]).call()
    assert staking_balance == 50
