// Wallet Service Hook - Compute Credits and Tokenomics
import { useState, useEffect, useCallback } from 'react';
import { ComputeCredit, FogNode, WalletState } from '../types';

interface WalletTransaction {
  id: string;
  type: 'credit_earned' | 'credit_spent' | 'fog_contribution' | 'peer_reward';
  amount: number;
  description: string;
  timestamp: Date;
  status: 'pending' | 'confirmed' | 'failed';
  relatedEntity?: string; // peer ID, fog node ID, etc.
}

interface StakingInfo {
  stakedAmount: number;
  stakingRewards: number;
  unstakingPeriod: number; // days
  currentAPR: number;
}

export interface WalletServiceHook {
  walletState: WalletState;
  stakingInfo: StakingInfo;
  earnCredits: (amount: number, description: string) => Promise<boolean>;
  spendCredits: (amount: number, description: string) => Promise<boolean>;
  transferCredits: (recipientId: string, amount: number) => Promise<boolean>;
  contributeFogResources: (resources: Partial<FogNode>) => Promise<boolean>;
  stakeCredits: (amount: number) => Promise<boolean>;
  unstakeCredits: (amount: number) => Promise<boolean>;
  getTransactionHistory: (limit?: number) => WalletTransaction[];
  refreshBalance: () => Promise<void>;
  calculatePeerReward: (dataSize: number, distance: number) => number;
}

export const useWalletService = (userId: string): WalletServiceHook => {
  const [walletState, setWalletState] = useState<WalletState>({
    balance: 1000, // Starting credits for demo
    transactions: [],
    fogContributions: [],
    isLoading: false
  });

  const [stakingInfo, setStakingInfo] = useState<StakingInfo>({
    stakedAmount: 0,
    stakingRewards: 0,
    unstakingPeriod: 7,
    currentAPR: 12.5
  });

  const [transactionHistory, setTransactionHistory] = useState<WalletTransaction[]>([]);

  // Initialize wallet and load existing data
  useEffect(() => {
    initializeWallet();
    startStakingRewardsCalculation();
  }, [userId]);

  const initializeWallet = async (): Promise<void> => {
    setWalletState(prev => ({ ...prev, isLoading: true }));

    try {
      // In production, load from secure storage/blockchain
      const existingBalance = localStorage.getItem(`wallet_balance_${userId}`);
      const existingTransactions = localStorage.getItem(`wallet_transactions_${userId}`);
      const existingStaking = localStorage.getItem(`wallet_staking_${userId}`);

      if (existingBalance) {
        setWalletState(prev => ({
          ...prev,
          balance: parseFloat(existingBalance),
          isLoading: false
        }));
      }

      if (existingTransactions) {
        setTransactionHistory(JSON.parse(existingTransactions));
      }

      if (existingStaking) {
        setStakingInfo(prev => ({ ...prev, ...JSON.parse(existingStaking) }));
      }

      // Initialize with some demo transactions
      if (!existingTransactions) {
        const demoTransactions: WalletTransaction[] = [
          {
            id: 'tx-001',
            type: 'credit_earned',
            amount: 500,
            description: 'Initial wallet setup bonus',
            timestamp: new Date(Date.now() - 86400000), // 1 day ago
            status: 'confirmed'
          },
          {
            id: 'tx-002',
            type: 'fog_contribution',
            amount: 200,
            description: 'Fog node uptime reward',
            timestamp: new Date(Date.now() - 43200000), // 12 hours ago
            status: 'confirmed',
            relatedEntity: 'fog-node-001'
          }
        ];
        setTransactionHistory(demoTransactions);
      }
    } catch (error) {
      console.error('Failed to initialize wallet:', error);
    } finally {
      setWalletState(prev => ({ ...prev, isLoading: false }));
    }
  };

  const startStakingRewardsCalculation = (): void => {
    // Calculate staking rewards every minute
    setInterval(() => {
      setStakingInfo(prev => {
        if (prev.stakedAmount > 0) {
          const rewardIncrement = (prev.stakedAmount * prev.currentAPR) / (365 * 24 * 60); // Per minute
          return {
            ...prev,
            stakingRewards: prev.stakingRewards + rewardIncrement
          };
        }
        return prev;
      });
    }, 60000);
  };

  const earnCredits = useCallback(async (amount: number, description: string): Promise<boolean> => {
    try {
      const transaction: WalletTransaction = {
        id: `tx-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
        type: 'credit_earned',
        amount,
        description,
        timestamp: new Date(),
        status: 'pending'
      };

      // Add to transaction history
      setTransactionHistory(prev => [transaction, ...prev]);

      // Simulate network confirmation delay
      setTimeout(() => {
        setTransactionHistory(prev =>
          prev.map(tx =>
            tx.id === transaction.id ? { ...tx, status: 'confirmed' } : tx
          )
        );

        // Update balance
        setWalletState(prev => {
          const newBalance = prev.balance + amount;
          localStorage.setItem(`wallet_balance_${userId}`, newBalance.toString());
          return { ...prev, balance: newBalance };
        });

        // Update transaction storage
        const updatedTransactions = transactionHistory.map(tx =>
          tx.id === transaction.id ? { ...tx, status: 'confirmed' } : tx
        );
        localStorage.setItem(`wallet_transactions_${userId}`, JSON.stringify(updatedTransactions));
      }, 2000);

      return true;
    } catch (error) {
      console.error('Failed to earn credits:', error);
      return false;
    }
  }, [userId, transactionHistory]);

  const spendCredits = useCallback(async (amount: number, description: string): Promise<boolean> => {
    if (walletState.balance < amount) {
      console.error('Insufficient balance');
      return false;
    }

    try {
      const transaction: WalletTransaction = {
        id: `tx-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
        type: 'credit_spent',
        amount: -amount,
        description,
        timestamp: new Date(),
        status: 'pending'
      };

      setTransactionHistory(prev => [transaction, ...prev]);

      // Process transaction
      setTimeout(() => {
        setTransactionHistory(prev =>
          prev.map(tx =>
            tx.id === transaction.id ? { ...tx, status: 'confirmed' } : tx
          )
        );

        setWalletState(prev => {
          const newBalance = prev.balance - amount;
          localStorage.setItem(`wallet_balance_${userId}`, newBalance.toString());
          return { ...prev, balance: newBalance };
        });
      }, 1500);

      return true;
    } catch (error) {
      console.error('Failed to spend credits:', error);
      return false;
    }
  }, [walletState.balance, userId]);

  const transferCredits = useCallback(async (recipientId: string, amount: number): Promise<boolean> => {
    if (walletState.balance < amount) {
      return false;
    }

    return await spendCredits(amount, `Transfer to ${recipientId}`);
  }, [walletState.balance, spendCredits]);

  const contributeFogResources = useCallback(async (resources: Partial<FogNode>): Promise<boolean> => {
    try {
      const fogNode: FogNode = {
        id: `fog-${Date.now()}`,
        name: `${userId}-fog-node`,
        location: 'Local Network',
        resources: {
          cpu: resources.resources?.cpu || 50,
          memory: resources.resources?.memory || 4096,
          storage: resources.resources?.storage || 10240,
          bandwidth: resources.resources?.bandwidth || 100
        },
        status: 'active',
        reputation: 100
      };

      setWalletState(prev => ({
        ...prev,
        fogContributions: [...prev.fogContributions, fogNode]
      }));

      // Earn credits for contributing resources
      const rewardAmount = calculateFogContributionReward(fogNode.resources);
      await earnCredits(rewardAmount, `Fog node contribution reward: ${fogNode.name}`);

      return true;
    } catch (error) {
      console.error('Failed to contribute fog resources:', error);
      return false;
    }
  }, [userId, earnCredits]);

  const stakeCredits = useCallback(async (amount: number): Promise<boolean> => {
    if (walletState.balance < amount) {
      return false;
    }

    try {
      const success = await spendCredits(amount, `Staking ${amount} credits`);
      if (success) {
        setStakingInfo(prev => {
          const newStakingInfo = {
            ...prev,
            stakedAmount: prev.stakedAmount + amount
          };
          localStorage.setItem(`wallet_staking_${userId}`, JSON.stringify(newStakingInfo));
          return newStakingInfo;
        });
      }
      return success;
    } catch (error) {
      console.error('Failed to stake credits:', error);
      return false;
    }
  }, [walletState.balance, spendCredits, userId]);

  const unstakeCredits = useCallback(async (amount: number): Promise<boolean> => {
    if (stakingInfo.stakedAmount < amount) {
      return false;
    }

    try {
      // In production, implement unstaking period
      setStakingInfo(prev => {
        const newStakingInfo = {
          ...prev,
          stakedAmount: prev.stakedAmount - amount
        };
        localStorage.setItem(`wallet_staking_${userId}`, JSON.stringify(newStakingInfo));
        return newStakingInfo;
      });

      await earnCredits(amount, `Unstaked ${amount} credits`);
      return true;
    } catch (error) {
      console.error('Failed to unstake credits:', error);
      return false;
    }
  }, [stakingInfo.stakedAmount, earnCredits, userId]);

  const getTransactionHistory = useCallback((limit = 50): WalletTransaction[] => {
    return transactionHistory.slice(0, limit);
  }, [transactionHistory]);

  const refreshBalance = useCallback(async (): Promise<void> => {
    setWalletState(prev => ({ ...prev, isLoading: true }));

    try {
      // In production, fetch from blockchain/server
      await new Promise(resolve => setTimeout(resolve, 1000));
    } finally {
      setWalletState(prev => ({ ...prev, isLoading: false }));
    }
  }, []);

  const calculatePeerReward = useCallback((dataSize: number, distance: number): number => {
    // Base reward + data size bonus + distance penalty
    const baseReward = 1;
    const sizeBonus = Math.min(dataSize / 1024, 5); // Max 5 credits for size
    const distancePenalty = Math.max(0, distance / 1000); // 1 credit penalty per km

    return Math.max(baseReward + sizeBonus - distancePenalty, 0.1);
  }, []);

  const calculateFogContributionReward = (resources: FogNode['resources']): number => {
    const cpuScore = resources.cpu / 100;
    const memoryScore = resources.memory / 8192;
    const storageScore = resources.storage / 102400;
    const bandwidthScore = resources.bandwidth / 1000;

    return Math.floor((cpuScore + memoryScore + storageScore + bandwidthScore) * 50);
  };

  return {
    walletState,
    stakingInfo,
    earnCredits,
    spendCredits,
    transferCredits,
    contributeFogResources,
    stakeCredits,
    unstakeCredits,
    getTransactionHistory,
    refreshBalance,
    calculatePeerReward
  };
};
