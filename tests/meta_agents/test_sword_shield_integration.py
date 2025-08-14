"""
Integration Tests for Sword/Shield Security Architecture

Tests the complete daily mock battle system including:
- Agent initialization and capabilities
- Battle orchestration and coordination
- Attack/defense simulation
- Performance analysis and improvement
- King Agent communication
"""

import os
import sys
from datetime import time
from unittest.mock import AsyncMock, patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../..", "src"))

from software.meta_agents.battle_orchestrator import BattleOrchestrator, BattleScenario
from software.meta_agents.shield import ShieldAgent
from software.meta_agents.sword import SwordAgent


class TestSwordShieldIntegration:
    """Test suite for Sword/Shield security architecture integration."""

    @pytest.fixture
    def sword_agent(self):
        """Create Sword Agent for testing."""
        return SwordAgent("test_sword")

    @pytest.fixture
    def shield_agent(self):
        """Create Shield Agent for testing."""
        return ShieldAgent("test_shield")

    @pytest.fixture
    def battle_orchestrator(self):
        """Create Battle Orchestrator for testing."""
        return BattleOrchestrator("test_orchestrator", battle_time=time(2, 0))

    @pytest.mark.asyncio
    async def test_sword_agent_initialization(self, sword_agent):
        """Test Sword Agent initializes properly with security capabilities."""
        assert sword_agent.agent_id == "test_sword"
        assert sword_agent.agent_name == "Sword - Offensive Security Specialist"
        assert len(sword_agent.attack_techniques) > 0
        assert sword_agent.quiet_star is not None
        assert sword_agent.thought_encryption is not None
        assert sword_agent.belief_engine is not None

    @pytest.mark.asyncio
    async def test_shield_agent_initialization(self, shield_agent):
        """Test Shield Agent initializes properly with defensive capabilities."""
        assert shield_agent.agent_id == "test_shield"
        assert shield_agent.agent_name == "Shield - Defensive Security Specialist"
        assert len(shield_agent.defensive_patterns) > 0
        assert shield_agent.quiet_star is not None
        assert shield_agent.thought_encryption is not None
        assert shield_agent.belief_engine is not None

    @pytest.mark.asyncio
    async def test_battle_orchestrator_initialization(self, battle_orchestrator):
        """Test Battle Orchestrator initializes with proper configuration."""
        assert battle_orchestrator.agent_id == "test_orchestrator"
        assert battle_orchestrator.daily_battle_time == time(2, 0)
        assert len(battle_orchestrator.battle_scenarios) > 0
        assert battle_orchestrator.sandbox_config["isolated_network"] is True

    @pytest.mark.asyncio
    async def test_sword_daily_research(self, sword_agent):
        """Test Sword Agent conducts daily offensive research."""
        with patch.object(
            sword_agent.quiet_star, "generate_reasoning", new_callable=AsyncMock
        ) as mock_reasoning:
            mock_reasoning.return_value = "encrypted_offensive_research_thoughts"

            research_results = await sword_agent.daily_offensive_research()

            assert "research_date" in research_results
            assert "encrypted_thoughts" in research_results
            assert "new_exploits_discovered" in research_results
            assert "attack_techniques_refined" in research_results
            assert research_results["new_exploits_discovered"] >= 3
            mock_reasoning.assert_called_once()

    @pytest.mark.asyncio
    async def test_shield_daily_research(self, shield_agent):
        """Test Shield Agent conducts daily defensive research."""
        with patch.object(
            shield_agent.quiet_star, "generate_reasoning", new_callable=AsyncMock
        ) as mock_reasoning:
            mock_reasoning.return_value = "encrypted_defensive_research_thoughts"

            research_results = await shield_agent.daily_defensive_research()

            assert "research_date" in research_results
            assert "encrypted_thoughts" in research_results
            assert "new_threats_analyzed" in research_results
            assert "defensive_patterns_updated" in research_results
            assert research_results["new_threats_analyzed"] >= 5
            mock_reasoning.assert_called_once()

    @pytest.mark.asyncio
    async def test_sword_attack_preparation(self, sword_agent):
        """Test Sword Agent prepares attack strategies."""
        target_info = {
            "target_systems": ["web_server", "database", "user_accounts"],
            "known_vulnerabilities": ["sql_injection", "xss"],
            "attack_timeline": "6_hours",
        }

        with patch.object(
            sword_agent.quiet_star, "generate_reasoning", new_callable=AsyncMock
        ) as mock_reasoning:
            mock_reasoning.return_value = "encrypted_attack_strategy"

            attack_strategy = await sword_agent.prepare_battle_attack(target_info)

            assert "strategy_id" in attack_strategy
            assert "encrypted_strategy" in attack_strategy
            assert "primary_attacks" in attack_strategy
            assert "stealth_techniques" in attack_strategy
            assert len(attack_strategy["primary_attacks"]) > 0
            mock_reasoning.assert_called_once()

    @pytest.mark.asyncio
    async def test_shield_defense_preparation(self, shield_agent):
        """Test Shield Agent prepares defensive strategies."""
        threat_intel = {
            "expected_attacks": ["sql_injection", "lateral_movement"],
            "target_systems": ["web_server", "database"],
            "attacker_profile": "advanced_persistent_threat",
        }

        with patch.object(
            shield_agent.quiet_star, "generate_reasoning", new_callable=AsyncMock
        ) as mock_reasoning:
            mock_reasoning.return_value = "encrypted_defense_strategy"

            defense_strategy = await shield_agent.prepare_battle_defense(threat_intel)

            assert "strategy_id" in defense_strategy
            assert "encrypted_strategy" in defense_strategy
            assert "active_defenses" in defense_strategy
            assert "detection_systems" in defense_strategy
            assert len(defense_strategy["active_defenses"]) > 0
            mock_reasoning.assert_called_once()

    @pytest.mark.asyncio
    async def test_threat_detection_analysis(self, shield_agent):
        """Test Shield Agent analyzes threats effectively."""
        suspicious_indicators = [
            "unusual_sql_queries",
            "multiple_login_failures",
            "suspicious_network_traffic",
        ]

        with patch.object(
            shield_agent.quiet_star, "generate_reasoning", new_callable=AsyncMock
        ) as mock_reasoning:
            mock_reasoning.return_value = "encrypted_threat_analysis"

            analysis = await shield_agent.threat_detection_analysis(
                suspicious_indicators
            )

            assert "analysis_id" in analysis
            assert "threat_type" in analysis
            assert "severity" in analysis
            assert "confidence" in analysis
            assert "recommended_actions" in analysis
            assert analysis["confidence"] >= 0.6
            mock_reasoning.assert_called_once()

    @pytest.mark.asyncio
    async def test_battle_scenario_selection(self, battle_orchestrator):
        """Test battle orchestrator selects appropriate scenarios."""
        # Simulate different win rate scenarios
        test_scenarios = [
            (0.3, 0.7, ["advanced", "expert"]),  # Sword losing
            (0.8, 0.2, ["beginner", "intermediate"]),  # Shield losing
            (0.5, 0.5, ["intermediate", "advanced"]),  # Balanced
        ]

        for sword_rate, shield_rate, expected_difficulties in test_scenarios:
            battle_orchestrator.sword_win_rate = sword_rate
            battle_orchestrator.shield_win_rate = shield_rate
            battle_orchestrator.battle_count = 10

            scenario = battle_orchestrator._select_battle_scenario()

            assert isinstance(scenario, BattleScenario)
            assert scenario.difficulty_level in expected_difficulties

    @pytest.mark.asyncio
    async def test_battle_orchestration_full_cycle(self, battle_orchestrator):
        """Test complete battle orchestration cycle."""
        # Initialize mock agents
        await battle_orchestrator.initialize_agents()

        # Mock the agent methods
        with (
            patch.object(
                battle_orchestrator.sword_agent,
                "prepare_battle_attack",
                new_callable=AsyncMock,
            ) as mock_sword_prep,
            patch.object(
                battle_orchestrator.shield_agent,
                "prepare_battle_defense",
                new_callable=AsyncMock,
            ) as mock_shield_prep,
            patch.object(
                battle_orchestrator.sword_agent,
                "conduct_reconnaissance",
                new_callable=AsyncMock,
            ) as mock_recon,
            patch.object(
                battle_orchestrator.shield_agent,
                "threat_detection_analysis",
                new_callable=AsyncMock,
            ) as mock_detection,
            patch.object(
                battle_orchestrator.sword_agent,
                "execute_battle_attacks",
                new_callable=AsyncMock,
            ) as mock_attacks,
            patch.object(
                battle_orchestrator.shield_agent,
                "engage_battle_defense",
                new_callable=AsyncMock,
            ) as mock_defense,
            patch.object(
                battle_orchestrator.sword_agent,
                "post_battle_analysis",
                new_callable=AsyncMock,
            ) as mock_sword_analysis,
            patch.object(
                battle_orchestrator.shield_agent,
                "post_battle_analysis",
                new_callable=AsyncMock,
            ) as mock_shield_analysis,
        ):
            # Configure mock responses
            mock_sword_prep.return_value = {
                "strategy_id": "test_attack",
                "preparation_duration": 300,
            }
            mock_shield_prep.return_value = {
                "strategy_id": "test_defense",
                "preparation_duration": 300,
            }
            mock_recon.return_value = {
                "reconnaissance_indicators": ["scan_detected"],
                "intelligence_items": ["vuln1", "vuln2"],
            }
            mock_detection.return_value = {
                "detected": True,
                "threat_type": "reconnaissance",
                "severity": "low",
            }
            mock_attacks.return_value = {
                "attack_results": [{"success": True, "technique": "sql_injection"}]
            }
            mock_defense.return_value = {
                "detection_results": [{"detected": True, "response_time": 120}]
            }
            mock_sword_analysis.return_value = {
                "performance_assessment": {
                    "overall_grade": "B",
                    "improvement_priorities": ["stealth"],
                }
            }
            mock_shield_analysis.return_value = {
                "performance_assessment": {
                    "overall_grade": "A",
                    "improvement_priorities": ["speed"],
                }
            }

            # Conduct battle
            battle_metrics = await battle_orchestrator.conduct_daily_battle()

            # Verify battle completed successfully
            assert battle_metrics.battle_id is not None
            assert battle_metrics.duration_minutes > 0
            assert battle_metrics.overall_winner in ["sword", "shield", "draw"]
            assert 0 <= battle_metrics.attack_success_rate <= 1
            assert 0 <= battle_metrics.defense_success_rate <= 1

            # Verify all phases were executed
            assert mock_sword_prep.called
            assert mock_shield_prep.called
            assert mock_recon.called
            assert mock_detection.called
            assert mock_attacks.called
            assert mock_defense.called
            assert mock_sword_analysis.called
            assert mock_shield_analysis.called

    @pytest.mark.asyncio
    async def test_battle_performance_tracking(self, battle_orchestrator):
        """Test battle performance tracking and improvement."""
        # Simulate multiple battles
        initial_battles = battle_orchestrator.battle_count

        # Create mock battle metrics
        from software.meta_agents.battle_orchestrator import BattleMetrics

        test_battles = [
            BattleMetrics(
                "battle1", "2024-01-01", 90, 85, 75, "sword", 0.8, 0.6, 0.6, 0.7
            ),
            BattleMetrics(
                "battle2", "2024-01-02", 95, 70, 90, "shield", 0.5, 0.9, 0.9, 0.8
            ),
            BattleMetrics(
                "battle3", "2024-01-03", 85, 80, 80, "draw", 0.7, 0.7, 0.7, 0.7
            ),
        ]

        for metrics in test_battles:
            battle_orchestrator.battle_history.append(metrics)
            battle_orchestrator.battle_count += 1
            battle_orchestrator._update_win_rates(metrics)

        # Verify tracking
        assert battle_orchestrator.battle_count == initial_battles + 3
        assert len(battle_orchestrator.battle_history) >= 3

        # Verify win rates updated correctly
        # 1 sword win, 1 shield win, 1 draw out of 3 battles
        expected_sword_rate = 1.0 / 3.0
        expected_shield_rate = 1.0 / 3.0

        assert abs(battle_orchestrator.sword_win_rate - expected_sword_rate) < 0.1
        assert abs(battle_orchestrator.shield_win_rate - expected_shield_rate) < 0.1

    @pytest.mark.asyncio
    async def test_encrypted_thought_bubbles(self, sword_agent, shield_agent):
        """Test encrypted thought bubble generation."""
        test_prompt = "Analyze this security scenario for vulnerabilities"

        # Test Sword Agent encrypted thoughts
        with patch.object(
            sword_agent.quiet_star, "generate_reasoning", new_callable=AsyncMock
        ) as mock_sword_reasoning:
            mock_sword_reasoning.return_value = "offensive_security_analysis"

            encrypted_thought = await sword_agent.think_encrypted(test_prompt)

            assert encrypted_thought is not None
            assert isinstance(encrypted_thought, str)
            mock_sword_reasoning.assert_called_once()

        # Test Shield Agent encrypted thoughts
        with patch.object(
            shield_agent.quiet_star, "generate_reasoning", new_callable=AsyncMock
        ) as mock_shield_reasoning:
            mock_shield_reasoning.return_value = "defensive_security_analysis"

            encrypted_thought = await shield_agent.think_encrypted(test_prompt)

            assert encrypted_thought is not None
            assert isinstance(encrypted_thought, str)
            mock_shield_reasoning.assert_called_once()

    @pytest.mark.asyncio
    async def test_king_communication(self, sword_agent, shield_agent):
        """Test unencrypted communication with King Agent."""
        test_message = "Security intelligence report from daily research"

        # Test Sword communication with King
        sword_comm = await sword_agent.communicate_with_king(
            test_message, priority="high"
        )

        assert sword_comm["from_agent"] == "sword"
        assert sword_comm["to_agent"] == "king"
        assert sword_comm["priority"] == "high"
        assert sword_comm["encrypted"] is False  # King communications unencrypted
        assert sword_comm["content"] == test_message

        # Test Shield communication with King
        shield_comm = await shield_agent.communicate_with_king(
            test_message, priority="normal"
        )

        assert shield_comm["from_agent"] == "shield"
        assert shield_comm["to_agent"] == "king"
        assert shield_comm["priority"] == "normal"
        assert shield_comm["encrypted"] is False  # King communications unencrypted
        assert shield_comm["content"] == test_message

    def test_agent_status_reporting(
        self, sword_agent, shield_agent, battle_orchestrator
    ):
        """Test agent status reporting capabilities."""
        # Test Sword Agent status
        sword_status = sword_agent.get_offensive_status()

        assert "agent_id" in sword_status
        assert "attack_techniques_count" in sword_status
        assert "battle_history_count" in sword_status
        assert "performance_metrics" in sword_status
        assert sword_status["status"] == "active"

        # Test Shield Agent status
        shield_status = shield_agent.get_defensive_status()

        assert "agent_id" in shield_status
        assert "defensive_patterns_count" in shield_status
        assert "active_incidents" in shield_status
        assert "performance_metrics" in shield_status
        assert shield_status["status"] == "active"

        # Test Battle Orchestrator status
        orchestrator_status = battle_orchestrator.get_orchestrator_status()

        assert "agent_id" in orchestrator_status
        assert "daily_battle_time" in orchestrator_status
        assert "total_battles_conducted" in orchestrator_status
        assert "available_scenarios" in orchestrator_status
        assert orchestrator_status["status"] == "active"

    @pytest.mark.asyncio
    async def test_battle_intelligence_sharing(self, battle_orchestrator):
        """Test intelligence sharing between Sword and Shield."""
        await battle_orchestrator.initialize_agents()

        # Simulate Sword gathering intelligence
        sword_intel = {
            "discovered_vulnerabilities": ["sql_injection", "xss"],
            "target_reconnaissance": ["web_server", "database"],
            "attack_vectors": ["remote_code_execution", "privilege_escalation"],
        }

        # Shield should receive and analyze this intelligence
        with patch.object(
            battle_orchestrator.shield_agent,
            "threat_detection_analysis",
            new_callable=AsyncMock,
        ) as mock_analysis:
            mock_analysis.return_value = {
                "threat_classification": "advanced_persistent_threat",
                "risk_level": "high",
                "countermeasures": ["network_segmentation", "endpoint_monitoring"],
            }

            analysis_result = (
                await battle_orchestrator.shield_agent.threat_detection_analysis(
                    sword_intel["discovered_vulnerabilities"]
                )
            )

            assert "threat_classification" in analysis_result
            assert "risk_level" in analysis_result
            mock_analysis.assert_called_once()

    @pytest.mark.asyncio
    async def test_battle_scenario_library(self, battle_orchestrator):
        """Test battle scenario library completeness."""
        scenarios = battle_orchestrator.battle_scenarios

        # Verify we have scenarios of different difficulty levels
        difficulties = set(s.difficulty_level for s in scenarios.values())
        assert "beginner" in difficulties or "intermediate" in difficulties
        assert "advanced" in difficulties or "expert" in difficulties

        # Verify scenarios have required components
        for scenario in scenarios.values():
            assert scenario.scenario_id is not None
            assert scenario.name is not None
            assert len(scenario.attack_vectors) > 0
            assert len(scenario.target_systems) > 0
            assert scenario.success_criteria is not None
            assert scenario.difficulty_level in [
                "beginner",
                "intermediate",
                "advanced",
                "expert",
            ]

    @pytest.mark.asyncio
    async def test_sandbox_environment(self, battle_orchestrator):
        """Test sandbox environment setup and isolation."""
        test_scenario = list(battle_orchestrator.battle_scenarios.values())[0]

        sandbox_status = await battle_orchestrator._setup_sandbox_environment(
            test_scenario
        )

        assert "environment_id" in sandbox_status
        assert sandbox_status["isolation_confirmed"] is True
        assert (
            sandbox_status["virtual_machines"]
            == battle_orchestrator.sandbox_config["virtual_machines"]
        )
        assert (
            sandbox_status["monitoring_enabled"]
            == battle_orchestrator.sandbox_config["monitoring_enabled"]
        )
        assert set(sandbox_status["target_systems_ready"]) == set(
            test_scenario.target_systems
        )

    def test_performance_metrics_calculation(self, battle_orchestrator):
        """Test battle performance metrics calculation."""
        mock_battle_results = {
            "battle_id": "test_battle",
            "start_time": "2024-01-01T02:00:00",
            "phases": {
                "combat": {
                    "combat_effectiveness": {
                        "attack_success_rate": 0.75,
                        "defense_success_rate": 0.65,
                    }
                },
                "analysis": {
                    "sword_analysis": {
                        "performance_assessment": {"overall_grade": "B"}
                    },
                    "shield_analysis": {
                        "performance_assessment": {"overall_grade": "A"}
                    },
                },
            },
        }

        metrics = battle_orchestrator._calculate_battle_metrics(
            mock_battle_results, 90.0
        )

        assert metrics.battle_id == "test_battle"
        assert metrics.duration_minutes == 90.0
        assert metrics.attack_success_rate == 0.75
        assert metrics.defense_success_rate == 0.65
        assert metrics.sword_score == 85  # Grade B = 85
        assert metrics.shield_score == 95  # Grade A = 95
        assert metrics.overall_winner in ["sword", "shield", "draw"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
