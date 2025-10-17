require("@nomiclabs/hardhat-waffle");
require("@nomiclabs/hardhat-etherscan");
require("hardhat-deploy");
require("hardhat-gas-reporter");
require("solidity-coverage");

const {
    PRIVATE_KEY,
    POLYGONSCAN_API_KEY,
    MUMBAI_RPC_URL,
    MAINNET_RPC_URL
} = process.env;

module.exports = {
    solidity: {
        version: "0.8.19",
        settings: {
            optimizer: {
                enabled: true,
                runs: 200
            }
        }
    },
    networks: {
        hardhat: {
            chainId: 1337,
            allowUnlimitedContractSize: false,
        },
        localhost: {
            url: "http://127.0.0.1:8545",
            chainId: 1337,
        },
        mumbai: {
            url: MUMBAI_RPC_URL || "https://rpc-mumbai.maticvigil.com",
            accounts: PRIVATE_KEY ? [PRIVATE_KEY] : [],
            chainId: 80001,
            gas: 2100000,
            gasPrice: 8000000000,
        },
        polygon: {
            url: MAINNET_RPC_URL || "https://polygon-rpc.com",
            accounts: PRIVATE_KEY ? [PRIVATE_KEY] : [],
            chainId: 137,
            gas: 2100000,
            gasPrice: 300000000000,
        }
    },
    etherscan: {
        apiKey: POLYGONSCAN_API_KEY
    },
    gasReporter: {
        enabled: process.env.REPORT_GAS ? true : false,
        currency: "USD",
        gasPrice: 100
    },
    namedAccounts: {
        deployer: {
            default: 0,
        },
    },
    paths: {
        sources: "./contracts",
        tests: "./tests",
        cache: "./cache",
        artifacts: "./artifacts"
    }
};