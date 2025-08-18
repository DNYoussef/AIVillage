// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

import "@openzeppelin/contracts/token/ERC20/ERC20.sol";
import "@openzeppelin/contracts/access/AccessControl.sol";

interface IEducationPlatform {
    function unlockPremium(address student) external;
}

contract VILLAGEToken is ERC20, AccessControl {
    bytes32 public constant MINTER_ROLE = keccak256("MINTER_ROLE");
    bytes32 public constant BURNER_ROLE = keccak256("BURNER_ROLE");

    mapping(address => bool) public isEducational;
    mapping(address => uint256) public stakingBalance;

    uint256 public constant MAX_SUPPLY = 1_000_000_000 * 10 ** 18;
    uint256 public burnRate = 3;
    address public educationContract;

    constructor() ERC20("VILLAGE", "VLG") {
        _grantRole(DEFAULT_ADMIN_ROLE, msg.sender);
        _grantRole(MINTER_ROLE, msg.sender);
    }

    function mintForEducation(address student, uint256 amount) external {
        require(hasRole(MINTER_ROLE, msg.sender), "Not authorized");
        require(totalSupply() + amount <= MAX_SUPPLY, "Max supply exceeded");
        _mint(student, amount);
        isEducational[student] = true;
    }

    function transfer(address to, uint256 amount) public override returns (bool) {
        uint256 burnAmount = (amount * burnRate) / 100;
        uint256 transferAmount = amount - burnAmount;
        _transfer(msg.sender, address(this), burnAmount);
        _transfer(msg.sender, to, transferAmount);
        return true;
    }

    function stakeForLearning(uint256 amount) external {
        require(balanceOf(msg.sender) >= amount, "Insufficient balance");
        _transfer(msg.sender, address(this), amount);
        stakingBalance[msg.sender] += amount;
        IEducationPlatform(educationContract).unlockPremium(msg.sender);
    }

    function setEducationContract(address contractAddr) external {
        require(hasRole(DEFAULT_ADMIN_ROLE, msg.sender), "Not authorized");
        educationContract = contractAddr;
    }
}
