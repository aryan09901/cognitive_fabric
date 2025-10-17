const { ethers } = require("hardhat");
const fs = require("fs");
const path = require("path");

async function main() {
    console.log("🚀 Starting Cognitive Fabric deployment...");

    const [deployer] = await ethers.getSigners();
    console.log(`Deploying contracts with account: ${deployer.address}`);
    console.log(`Account balance: ${(await deployer.getBalance()).toString()}`);

    // Deploy KnowledgeToken
    console.log("\n📦 Deploying KnowledgeToken...");
    const KnowledgeToken = await ethers.getContractFactory("KnowledgeToken");
    const knowledgeToken = await KnowledgeToken.deploy();
    await knowledgeToken.deployed();
    console.log(`KnowledgeToken deployed to: ${knowledgeToken.address}`);

    // Deploy ReputationSystem
    console.log("\n🏆 Deploying ReputationSystem...");
    const ReputationSystem = await ethers.getContractFactory("ReputationSystem");
    const reputationSystem = await ReputationSystem.deploy();
    await reputationSystem.deployed();
    console.log(`ReputationSystem deployed to: ${reputationSystem.address}`);

    // Deploy CognitiveFabric
    console.log("\n🧠 Deploying CognitiveFabric...");
    const CognitiveFabric = await ethers.getContractFactory("CognitiveFabric");
    const cognitiveFabric = await CognitiveFabric.deploy();
    await cognitiveFabric.deployed();
    console.log(`CognitiveFabric deployed to: ${cognitiveFabric.address}`);

    // Set up contract relationships
    console.log("\n🔗 Setting up contract relationships...");

    // Update KnowledgeToken with CognitiveFabric address
    console.log("Updating KnowledgeToken fabric contract...");
    await knowledgeToken.updateFabricContract(cognitiveFabric.address);

    // Whitelist CognitiveFabric in ReputationSystem
    console.log("Whitelisting CognitiveFabric in ReputationSystem...");
    await reputationSystem.whitelistContract(cognitiveFabric.address);

    // Save deployment info
    const deploymentInfo = {
        network: network.name,
        timestamp: new Date().toISOString(),
        contracts: {
            KnowledgeToken: knowledgeToken.address,
            ReputationSystem: reputationSystem.address,
            CognitiveFabric: cognitiveFabric.address
        },
        deployer: deployer.address
    };

    // Create artifacts directory if it doesn't exist
    const artifactsDir = path.join(__dirname, "..", "artifacts");
    if (!fs.existsSync(artifactsDir)) {
        fs.mkdirSync(artifactsDir, { recursive: true });
    }

    // Save deployment info to file
    const deploymentPath = path.join(artifactsDir, "deployment.json");
    fs.writeFileSync(deploymentPath, JSON.stringify(deploymentInfo, null, 2));

    console.log("\n✅ Deployment completed successfully!");
    console.log(`📄 Deployment info saved to: ${deploymentPath}`);

    console.log("\n📋 Contract Addresses:");
    console.log(`   KnowledgeToken: ${knowledgeToken.address}`);
    console.log(`   ReputationSystem: ${reputationSystem.address}`);
    console.log(`   CognitiveFabric: ${cognitiveFabric.address}`);

    // Verify on PolygonScan (if on testnet/mainnet)
    if (network.name !== "hardhat" && network.name !== "localhost") {
        console.log("\n🔍 Verifying contracts on PolygonScan...");

        // Wait for blocks to be mined
        console.log("Waiting for block confirmations...");
        await cognitiveFabric.deployTransaction.wait(6);

        // Verify contracts
        await run("verify:verify", {
            address: knowledgeToken.address,
            constructorArguments: [],
        });

        await run("verify:verify", {
            address: reputationSystem.address,
            constructorArguments: [],
        });

        await run("verify:verify", {
            address: cognitiveFabric.address,
            constructorArguments: [],
        });

        console.log("✅ Contracts verified on PolygonScan!");
    }
}

main()
    .then(() => process.exit(0))
    .catch((error) => {
        console.error("❌ Deployment failed:", error);
        process.exit(1);
    });