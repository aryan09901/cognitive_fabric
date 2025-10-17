// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

import "@openzeppelin/contracts/access/Ownable.sol";

/**
 * @title ReputationSystem
 * @dev Advanced reputation system for Cognitive Fabric agents
 */
contract ReputationSystem is Ownable {
    // Structs
    struct ReputationData {
        uint256 score;
        uint256 totalInteractions;
        uint256 positiveInteractions;
        uint256 knowledgeContributions;
        uint256 lastUpdate;
        uint256 streak; // Consecutive positive interactions
    }
    
    // Mappings
    mapping(address => ReputationData) public reputation;
    mapping(address => bool) public whitelistedContracts;
    
    // Constants
    uint256 public constant INITIAL_REPUTATION = 100;
    uint256 public constant MAX_REPUTATION = 1000;
    uint256 public constant POSITIVE_INTERACTION_BONUS = 5;
    uint256 public constant NEGATIVE_INTERACTION_PENALTY = 10;
    uint256 public constant KNOWLEDGE_CONTRIBUTION_BONUS = 8;
    uint256 public constant STREAK_BONUS = 2;
    
    // Events
    event ReputationUpdated(address indexed agent, uint256 newScore, int256 change, string reason);
    event ContractWhitelisted(address indexed contractAddress);
    event ContractBlacklisted(address indexed contractAddress);
    
    // Modifiers
    modifier onlyWhitelisted() {
        require(whitelistedContracts[msg.sender], "Caller not whitelisted");
        _;
    }
    
    constructor() {
        whitelistedContracts[msg.sender] = true;
    }
    
    /**
     * @dev Initialize reputation for a new agent
     * @param agent Address of the agent
     */
    function initializeReputation(address agent) external onlyWhitelisted {
        require(reputation[agent].lastUpdate == 0, "Reputation already initialized");
        
        reputation[agent] = ReputationData({
            score: INITIAL_REPUTATION,
            totalInteractions: 0,
            positiveInteractions: 0,
            knowledgeContributions: 0,
            lastUpdate: block.timestamp,
            streak: 0
        });
        
        emit ReputationUpdated(agent, INITIAL_REPUTATION, int256(INITIAL_REPUTATION), "Initialization");
    }
    
    /**
     * @dev Update reputation based on interaction outcome
     * @param agent Address of the agent
     * @param satisfactionScore Satisfaction score (0-100)
     * @param knowledgeShared Whether knowledge was shared
     */
    function updateReputation(
        address agent,
        uint256 satisfactionScore,
        bool knowledgeShared
    ) external onlyWhitelisted {
        require(reputation[agent].lastUpdate > 0, "Reputation not initialized");
        
        ReputationData storage rep = reputation[agent];
        uint256 oldScore = rep.score;
        
        // Update interaction counts
        rep.totalInteractions++;
        rep.lastUpdate = block.timestamp;
        
        if (satisfactionScore >= 80) {
            // Positive interaction
            rep.positiveInteractions++;
            rep.streak++;
            
            // Base bonus for positive interaction
            uint256 bonus = POSITIVE_INTERACTION_BONUS;
            
            // Streak bonus
            if (rep.streak > 1) {
                bonus += (rep.streak - 1) * STREAK_BONUS;
            }
            
            rep.score = _min(MAX_REPUTATION, rep.score + bonus);
            
        } else if (satisfactionScore <= 30) {
            // Negative interaction
            rep.streak = 0; // Reset streak
            rep.score = rep.score > NEGATIVE_INTERACTION_PENALTY ? 
                rep.score - NEGATIVE_INTERACTION_PENALTY : 0;
        }
        
        // Bonus for knowledge sharing
        if (knowledgeShared) {
            rep.knowledgeContributions++;
            rep.score = _min(MAX_REPUTATION, rep.score + KNOWLEDGE_CONTRIBUTION_BONUS);
        }
        
        // Decay over time (1 point per week)
        uint256 weeksSinceUpdate = (block.timestamp - rep.lastUpdate) / 1 weeks;
        if (weeksSinceUpdate > 0) {
            rep.score = rep.score > weeksSinceUpdate ? rep.score - weeksSinceUpdate : 0;
        }
        
        emit ReputationUpdated(agent, rep.score, int256(rep.score) - int256(oldScore), "Interaction Update");
    }
    
    /**
     * @dev Get comprehensive reputation data
     * @param agent Address of the agent
     */
    function getReputationData(address agent) external view returns (ReputationData memory) {
        return reputation[agent];
    }
    
    /**
     * @dev Calculate trust score (0-100) based on reputation
     * @param agent Address of the agent
     */
    function calculateTrustScore(address agent) external view returns (uint256) {
        ReputationData memory rep = reputation[agent];
        if (rep.totalInteractions == 0) return 50; // Default trust for new agents
        
        // Base trust calculation
        uint256 baseTrust = (rep.positiveInteractions * 100) / rep.totalInteractions;
        
        // Factor in reputation score
        uint256 reputationFactor = (rep.score * 100) / MAX_REPUTATION;
        
        // Factor in knowledge contributions
        uint256 knowledgeFactor = _min(100, rep.knowledgeContributions * 5);
        
        // Weighted average
        return (baseTrust * 40 + reputationFactor * 40 + knowledgeFactor * 20) / 100;
    }
    
    /**
     * @dev Whitelist a contract to call reputation functions
     * @param contractAddress Address to whitelist
     */
    function whitelistContract(address contractAddress) external onlyOwner {
        whitelistedContracts[contractAddress] = true;
        emit ContractWhitelisted(contractAddress);
    }
    
    /**
     * @dev Remove contract from whitelist
     * @param contractAddress Address to remove
     */
    function blacklistContract(address contractAddress) external onlyOwner {
        whitelistedContracts[contractAddress] = false;
        emit ContractBlacklisted(contractAddress);
    }
    
    /**
     * @dev Internal function to calculate minimum
     */
    function _min(uint256 a, uint256 b) internal pure returns (uint256) {
        return a < b ? a : b;
    }
}