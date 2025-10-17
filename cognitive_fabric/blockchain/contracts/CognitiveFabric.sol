// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

import "@openzeppelin/contracts/security/ReentrancyGuard.sol";
import "@openzeppelin/contracts/access/Ownable.sol";

/**
 * @title Cognitive Fabric Main Contract
 * @dev Core contract for agent registry, reputation, and knowledge economy
 */
contract CognitiveFabric is ReentrancyGuard, Ownable {
    // Structs
    struct Agent {
        address owner;
        string metadata; // IPFS hash of agent config
        uint256 reputation;
        uint256 knowledgeContributed;
        uint256 tokensEarned;
        uint256 createdAt;
        bool registered;
        bool active;
    }
    
    struct KnowledgeItem {
        address contributor;
        string contentHash; // IPFS hash of knowledge
        string metadata; // IPFS hash of metadata
        uint256 timestamp;
        uint256 verificationScore;
        uint256 usefulnessScore;
        bool verified;
    }
    
    struct Interaction {
        address fromAgent;
        address toAgent;
        string queryHash;
        string responseHash;
        uint256 timestamp;
        uint256 satisfactionScore;
    }
    
    // Mappings
    mapping(address => Agent) public agents;
    mapping(string => KnowledgeItem) public knowledgeItems;
    mapping(bytes32 => Interaction) public interactions;
    
    // Arrays for iteration
    address[] public agentAddresses;
    string[] public knowledgeHashes;
    
    // Events
    event AgentRegistered(address indexed agentAddress, address owner, string metadata);
    event AgentReputationUpdated(address indexed agentAddress, uint256 newReputation, uint256 change);
    event KnowledgeShared(address indexed contributor, string knowledgeHash, uint256 reward);
    knowledgeItemVerified(string indexed knowledgeHash, uint256 verificationScore);
    event InteractionRecorded(bytes32 interactionId, address fromAgent, address toAgent, uint256 satisfaction);
    
    // Constants
    uint256 public constant INITIAL_REPUTATION = 100;
    uint256 public constant KNOWLEDGE_REWARD = 5;
    uint256 public constant VERIFICATION_THRESHOLD = 70; // 70% verification score
    
    // Modifiers
    modifier onlyRegisteredAgent() {
        require(agents[msg.sender].registered, "Agent not registered");
        _;
    }
    
    modifier onlyActiveAgent() {
        require(agents[msg.sender].active, "Agent not active");
        _;
    }
    
    /**
     * @dev Register a new AI agent
     * @param _metadata IPFS hash of agent configuration
     */
    function registerAgent(string memory _metadata) external {
        require(!agents[msg.sender].registered, "Agent already registered");
        require(bytes(_metadata).length > 0, "Metadata cannot be empty");
        
        agents[msg.sender] = Agent({
            owner: msg.sender,
            metadata: _metadata,
            reputation: INITIAL_REPUTATION,
            knowledgeContributed: 0,
            tokensEarned: 0,
            createdAt: block.timestamp,
            registered: true,
            active: true
        });
        
        agentAddresses.push(msg.sender);
        emit AgentRegistered(msg.sender, msg.sender, _metadata);
    }
    
    /**
     * @dev Share knowledge to the network
     * @param _knowledgeHash IPFS hash of knowledge content
     * @param _metadata IPFS hash of knowledge metadata
     */
    function shareKnowledge(
        string memory _knowledgeHash, 
        string memory _metadata
    ) external onlyRegisteredAgent onlyActiveAgent nonReentrant {
        require(!knowledgeExists(_knowledgeHash), "Knowledge already exists");
        
        // Create knowledge item
        knowledgeItems[_knowledgeHash] = KnowledgeItem({
            contributor: msg.sender,
            contentHash: _knowledgeHash,
            metadata: _metadata,
            timestamp: block.timestamp,
            verificationScore: 0,
            usefulnessScore: 0,
            verified: false
        });
        
        knowledgeHashes.push(_knowledgeHash);
        
        // Update agent stats
        agents[msg.sender].knowledgeContributed++;
        agents[msg.sender].reputation += KNOWLEDGE_REWARD;
        agents[msg.sender].tokensEarned += KNOWLEDGE_REWARD;
        
        emit KnowledgeShared(msg.sender, _knowledgeHash, KNOWLEDGE_REWARD);
        emit AgentReputationUpdated(msg.sender, agents[msg.sender].reputation, KNOWLEDGE_REWARD);
    }
    
    /**
     * @dev Verify knowledge item
     * @param _knowledgeHash IPFS hash of knowledge to verify
     * @param _verificationScore Verification score (0-100)
     */
    function verifyKnowledge(
        string memory _knowledgeHash, 
        uint256 _verificationScore
    ) external onlyRegisteredAgent onlyActiveAgent {
        require(knowledgeExists(_knowledgeHash), "Knowledge does not exist");
        require(_verificationScore <= 100, "Invalid verification score");
        
        KnowledgeItem storage item = knowledgeItems[_knowledgeHash];
        item.verificationScore = _verificationScore;
        item.verified = _verificationScore >= VERIFICATION_THRESHOLD;
        
        // Reward verifier with reputation
        agents[msg.sender].reputation += 2;
        
        emit knowledgeItemVerified(_knowledgeHash, _verificationScore);
    }
    
    /**
     * @dev Record agent interaction
     * @param _toAgent Address of agent being queried
     * @param _queryHash IPFS hash of query
     * @param _responseHash IPFS hash of response
     * @param _satisfactionScore Satisfaction score (0-100)
     */
    function recordInteraction(
        address _toAgent,
        string memory _queryHash,
        string memory _responseHash,
        uint256 _satisfactionScore
    ) external onlyRegisteredAgent onlyActiveAgent returns (bytes32) {
        require(agents[_toAgent].registered, "Target agent not registered");
        require(_satisfactionScore <= 100, "Invalid satisfaction score");
        
        bytes32 interactionId = keccak256(abi.encodePacked(
            msg.sender, _toAgent, _queryHash, block.timestamp
        ));
        
        interactions[interactionId] = Interaction({
            fromAgent: msg.sender,
            toAgent: _toAgent,
            queryHash: _queryHash,
            responseHash: _responseHash,
            timestamp: block.timestamp,
            satisfactionScore: _satisfactionScore
        });
        
        // Update reputation based on satisfaction
        if (_satisfactionScore >= 80) {
            agents[_toAgent].reputation += 3;
        } else if (_satisfactionScore <= 30) {
            agents[_toAgent].reputation = agents[_toAgent].reputation > 5 ? 
                agents[_toAgent].reputation - 5 : 0;
        }
        
        emit InteractionRecorded(interactionId, msg.sender, _toAgent, _satisfactionScore);
        return interactionId;
    }
    
    /**
     * @dev Get agent reputation
     * @param _agent Address of agent
     */
    function getAgentReputation(address _agent) public view returns (uint256) {
        return agents[_agent].reputation;
    }
    
    /**
     * @dev Get knowledge item
     * @param _knowledgeHash IPFS hash of knowledge
     */
    function getKnowledgeItem(
        string memory _knowledgeHash
    ) public view returns (KnowledgeItem memory) {
        require(knowledgeExists(_knowledgeHash), "Knowledge does not exist");
        return knowledgeItems[_knowledgeHash];
    }
    
    /**
     * @dev Check if knowledge exists
     * @param _knowledgeHash IPFS hash to check
     */
    function knowledgeExists(string memory _knowledgeHash) public view returns (bool) {
        return bytes(knowledgeItems[_knowledgeHash].contentHash).length > 0;
    }
    
    /**
     * @dev Get all registered agents
     */
    function getAllAgents() public view returns (address[] memory) {
        return agentAddresses;
    }
    
    /**
     * @dev Get agent count
     */
    function getAgentCount() public view returns (uint256) {
        return agentAddresses.length;
    }
}