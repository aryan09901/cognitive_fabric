// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

import "@openzeppelin/contracts/token/ERC20/ERC20.sol";
import "@openzeppelin/contracts/token/ERC20/extensions/ERC20Burnable.sol";
import "@openzeppelin/contracts/access/Ownable.sol";
import "@openzeppelin/contracts/security/Pausable.sol";

/**
 * @title KnowledgeToken
 * @dev ERC20 token for incentivizing knowledge sharing in Cognitive Fabric
 */
contract KnowledgeToken is ERC20, ERC20Burnable, Ownable, Pausable {
    // Cognitive Fabric contract address
    address public cognitiveFabric;
    
    // Reward parameters
    uint256 public constant KNOWLEDGE_SHARE_REWARD = 10 * 10**18; // 10 tokens
    uint256 public constant VERIFICATION_REWARD = 5 * 10**18;     // 5 tokens
    uint256 public constant SATISFACTION_REWARD = 2 * 10**18;     // 2 tokens
    
    // Events
    event RewardsDistributed(address indexed agent, uint256 knowledgeReward, uint256 verificationReward, uint256 satisfactionReward);
    event FabricContractUpdated(address indexed newFabric);
    
    // Modifiers
    modifier onlyFabric() {
        require(msg.sender == cognitiveFabric, "Only Cognitive Fabric can call this");
        _;
    }
    
    constructor() ERC20("KnowledgeToken", "KNWL") {
        // Initial supply: 10 million tokens
        _mint(msg.sender, 10_000_000 * 10**18);
        cognitiveFabric = msg.sender;
    }
    
    /**
     * @dev Distribute rewards to an agent for various activities
     * @param agent Address of the agent to reward
     * @param knowledgeShared Whether knowledge was shared
     * @param verificationDone Whether verification was performed
     * @param highSatisfaction Whether interaction had high satisfaction
     */
    function distributeRewards(
        address agent,
        bool knowledgeShared,
        bool verificationDone,
        bool highSatisfaction
    ) external onlyFabric whenNotPaused {
        uint256 totalReward = 0;
        
        if (knowledgeShared) {
            totalReward += KNOWLEDGE_SHARE_REWARD;
        }
        
        if (verificationDone) {
            totalReward += VERIFICATION_REWARD;
        }
        
        if (highSatisfaction) {
            totalReward += SATISFACTION_REWARD;
        }
        
        if (totalReward > 0) {
            _mint(agent, totalReward);
            emit RewardsDistributed(agent, 
                knowledgeShared ? KNOWLEDGE_SHARE_REWARD : 0,
                verificationDone ? VERIFICATION_REWARD : 0,
                highSatisfaction ? SATISFACTION_REWARD : 0
            );
        }
    }
    
    /**
     * @dev Update Cognitive Fabric contract address
     * @param newFabric Address of new Cognitive Fabric contract
     */
    function updateFabricContract(address newFabric) external onlyOwner {
        require(newFabric != address(0), "Invalid address");
        cognitiveFabric = newFabric;
        emit FabricContractUpdated(newFabric);
    }
    
    /**
     * @dev Mint new tokens (only owner, for ecosystem growth)
     * @param to Address to mint to
     * @param amount Amount to mint
     */
    function ecosystemMint(address to, uint256 amount) external onlyOwner {
        _mint(to, amount);
    }
    
    /**
     * @dev Pause token transfers (emergency only)
     */
    function pause() external onlyOwner {
        _pause();
    }
    
    /**
     * @dev Unpause token transfers
     */
    function unpause() external onlyOwner {
        _unpause();
    }
    
    /**
     * @dev Hook that is called before any transfer of tokens
     */
    function _beforeTokenTransfer(address from, address to, uint256 amount)
        internal
        override
        whenNotPaused
    {
        super._beforeTokenTransfer(from, to, amount);
    }
}