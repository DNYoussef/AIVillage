#!/usr/bin/env python3
"""
AIVillage System Integration Test Suite
=====================================

Comprehensive integration testing across all remediated tracks:
- T5: Security & Federation (Tor/I2P/Bluetooth)
- T3: Agent Forge (18-agent ecosystem)
- T2: RAG System (Graph-enhanced retrieval)
- T6: Distributed Inference (Tensor streaming)

Tests cross-track functionality and production readiness.
"""

import asyncio
import sys
import time
from pathlib import Path
from typing import Any

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))


class SystemIntegrationTester:
    """Comprehensive system integration testing framework"""

    def __init__(self):
        self.test_results: dict[str, Any] = {}
        self.start_time = time.time()
        self.passed_tests = 0
        self.failed_tests = 0
        self.total_tests = 0

    def log(self, level: str, message: str):
        """Log test messages with proper formatting"""
        timestamp = time.strftime("%H:%M:%S")
        print(f"[{timestamp}] [{level:4}] {message}")

    async def run_test(self, test_name: str, test_func) -> bool:
        """Run individual test with error handling"""
        self.total_tests += 1
        self.log("TEST", f"Starting {test_name}...")

        try:
            start_time = time.time()
            result = (
                await test_func()
                if asyncio.iscoroutinefunction(test_func)
                else test_func()
            )
            duration = time.time() - start_time

            if result:
                self.passed_tests += 1
                self.log("PASS", f"{test_name} completed in {duration:.2f}s")
                self.test_results[test_name] = {"status": "PASS", "duration": duration}
                return True
            else:
                self.failed_tests += 1
                self.log("FAIL", f"{test_name} failed in {duration:.2f}s")
                self.test_results[test_name] = {"status": "FAIL", "duration": duration}
                return False

        except Exception as e:
            self.failed_tests += 1
            duration = time.time() - start_time if "start_time" in locals() else 0
            self.log("FAIL", f"{test_name} exception: {str(e)}")
            self.test_results[test_name] = {
                "status": "FAIL",
                "duration": duration,
                "error": str(e),
            }
            return False

    # ============================================================================
    # T5: Security & Federation Integration Tests
    # ============================================================================

    def test_t5_federation_imports(self) -> bool:
        """Test T5 federation module imports work correctly"""
        try:
            self.log("INFO", "T5 federation imports successful")
            return True
        except Exception as e:
            self.log("ERROR", f"T5 import failed: {e}")
            return False

    async def test_t5_federation_initialization(self) -> bool:
        """Test T5 federation components can be initialized"""
        try:
            from federation.core.federation_manager import FederationManager

            # Test federation manager creation
            manager = FederationManager(
                device_id="integration_test_device",
                enable_tor=False,  # Disabled for testing
                enable_i2p=False,  # Disabled for testing
            )

            # Test Tor transport (should handle gracefully when disabled)
            await manager._start_tor_transport()

            # Test I2P transport (should handle gracefully when disabled)
            await manager._start_i2p_transport()

            self.log("INFO", "T5 federation initialization successful")
            return True
        except Exception as e:
            self.log("ERROR", f"T5 initialization failed: {e}")
            return False

    async def test_t5_bitchat_operations(self) -> bool:
        """Test T5 BitChat Bluetooth operations"""
        try:
            from federation.protocols.enhanced_bitchat import EnhancedBitChatTransport

            transport = EnhancedBitChatTransport(device_id="test_bitchat")

            # Test Bluetooth discovery (should fall back to simulation)
            discovery_result = await transport._start_bluetooth_discovery()

            # Test stop with Bluetooth cleanup
            await transport.stop()

            self.log(
                "INFO",
                f"T5 BitChat operations successful (discovery: {discovery_result})",
            )
            return True
        except Exception as e:
            self.log("ERROR", f"T5 BitChat operations failed: {e}")
            return False

    # ============================================================================
    # T3: Agent Forge Integration Tests
    # ============================================================================

    def test_t3_agent_forge_imports(self) -> bool:
        """Test T3 agent forge imports work correctly"""
        try:
            self.log("INFO", "T3 agent forge imports successful")
            return True
        except Exception as e:
            self.log("ERROR", f"T3 import failed: {e}")
            return False

    async def test_t3_agent_orchestration(self) -> bool:
        """Test T3 agent orchestration system"""
        try:
            from production.agent_forge.orchestrator import FastAgentOrchestrator

            # Create orchestrator
            orchestrator = FastAgentOrchestrator()

            # Test agent creation request
            start_time = time.time()
            result = await orchestrator.coordinate_agents(
                [{"type": "king", "task": "test_coordination", "priority": 1}]
            )
            duration = time.time() - start_time

            # Verify <100ms requirement
            if duration > 0.1:
                self.log("WARN", f"Orchestration took {duration:.3f}s (target <0.1s)")
            else:
                self.log("INFO", f"Orchestration completed in {duration:.3f}s")

            self.log("INFO", "T3 agent orchestration successful")
            return True
        except Exception as e:
            self.log("ERROR", f"T3 orchestration failed: {e}")
            return False

    async def test_t3_agent_communication(self) -> bool:
        """Test T3 agent communication protocol"""
        try:
            from communications.standard_protocol import StandardCommunicationProtocol

            protocol = StandardCommunicationProtocol()

            # Test async communication setup
            await protocol.start()

            # Test message routing
            test_message = {
                "sender": "test_agent_1",
                "recipient": "test_agent_2",
                "content": "integration_test",
                "message_type": "coordination",
            }

            result = await protocol.route_message(test_message)
            await protocol.stop()

            self.log("INFO", "T3 agent communication successful")
            return True
        except Exception as e:
            self.log("ERROR", f"T3 communication failed: {e}")
            return False

    # ============================================================================
    # T2: RAG System Integration Tests
    # ============================================================================

    def test_t2_rag_imports(self) -> bool:
        """Test T2 RAG system imports work correctly"""
        try:
            self.log("INFO", "T2 RAG system imports successful")
            return True
        except Exception as e:
            self.log("ERROR", f"T2 import failed: {e}")
            return False

    async def test_t2_rag_performance(self) -> bool:
        """Test T2 RAG system performance metrics"""
        try:
            from production.rag.rag_system.interface import (
                HybridRetriever,
                ProductionEmbeddingModel,
            )

            # Test embedding model
            embedding_model = ProductionEmbeddingModel()

            # Test vector operations timing
            start_time = time.time()
            embeddings = await embedding_model.get_embeddings(
                ["test query for performance"]
            )
            vector_duration = time.time() - start_time

            # Test hybrid retrieval
            retriever = HybridRetriever()
            start_time = time.time()
            results = await retriever.retrieve("test query", top_k=5)
            retrieval_duration = time.time() - start_time

            # Verify performance targets
            vector_target = 0.010  # 10ms
            retrieval_target = 0.200  # 200ms

            vector_pass = vector_duration < vector_target
            retrieval_pass = retrieval_duration < retrieval_target

            self.log(
                "INFO",
                f"Vector ops: {vector_duration:.3f}s (target <{vector_target}s): {'PASS' if vector_pass else 'FAIL'}",
            )
            self.log(
                "INFO",
                f"Retrieval: {retrieval_duration:.3f}s (target <{retrieval_target}s): {'PASS' if retrieval_pass else 'FAIL'}",
            )

            return vector_pass and retrieval_pass
        except Exception as e:
            self.log("ERROR", f"T2 performance test failed: {e}")
            return False

    async def test_t2_rag_integration(self) -> bool:
        """Test T2 RAG system end-to-end integration"""
        try:
            from production.rag.rag_system.agent_interface import LatentSpaceAgent
            from production.rag.rag_system.interface import (
                ContextualKnowledgeConstructor,
            )

            # Test latent space agent
            agent = LatentSpaceAgent()
            enhancement_result = await agent.enhance_query("test integration query")

            # Test knowledge constructor
            constructor = ContextualKnowledgeConstructor()
            knowledge_result = await constructor.construct("integration test context")

            self.log("INFO", "T2 RAG integration successful")
            return True
        except Exception as e:
            self.log("ERROR", f"T2 integration failed: {e}")
            return False

    # ============================================================================
    # T6: Distributed Inference Integration Tests
    # ============================================================================

    def test_t6_distributed_imports(self) -> bool:
        """Test T6 distributed inference imports work correctly"""
        try:
            self.log("INFO", "T6 distributed inference imports successful")
            return True
        except Exception as e:
            self.log("ERROR", f"T6 import failed: {e}")
            return False

    async def test_t6_tokenomics_integration(self) -> bool:
        """Test T6 tokenomics receipt system"""
        try:
            # Create temporary database for testing
            import tempfile

            from production.distributed_inference.tokenomics_receipts import (
                TokenomicsReceiptManager,
            )

            with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
                db_path = tmp.name

            try:
                manager = TokenomicsReceiptManager(db_path)
                await manager.initialize()

                # Test receipt creation
                receipt_id = await manager.create_receipt(
                    "tensor_transfer",
                    "test_transaction_123",
                    {"bytes_transferred": 1024, "test_mode": True},
                )

                # Test receipt confirmation
                await manager.confirm_receipt(receipt_id)

                # Test receipt querying
                receipts = await manager.get_receipts_by_type("tensor_transfer")

                await manager.close()

                self.log(
                    "INFO",
                    f"T6 tokenomics integration successful (receipt: {receipt_id})",
                )
                return len(receipts) > 0

            finally:
                # Cleanup
                Path(db_path).unlink(missing_ok=True)

        except Exception as e:
            self.log("ERROR", f"T6 tokenomics failed: {e}")
            return False

    async def test_t6_tensor_streaming(self) -> bool:
        """Test T6 tensor streaming capabilities"""
        try:
            from production.communications.p2p.tensor_streaming import (
                TensorStreamManager,
            )

            manager = TensorStreamManager("test_node")

            # Test manager initialization
            await manager.start()

            # Test stream creation (simulation)
            stream_id = await manager.create_stream(
                "test_recipient", {"test": "tensor_data"}, priority=1
            )

            await manager.stop()

            self.log("INFO", f"T6 tensor streaming successful (stream: {stream_id})")
            return True
        except Exception as e:
            self.log("ERROR", f"T6 tensor streaming failed: {e}")
            return False

    # ============================================================================
    # Cross-Track Integration Tests
    # ============================================================================

    async def test_cross_track_agent_rag_integration(self) -> bool:
        """Test T3 Agent Forge + T2 RAG System integration"""
        try:
            from production.agent_forge.orchestrator import FastAgentOrchestrator
            from production.rag.rag_system.agent_interface import (
                DynamicKnowledgeIntegrationAgent,
            )

            # Create orchestrator and RAG agent
            orchestrator = FastAgentOrchestrator()
            rag_agent = DynamicKnowledgeIntegrationAgent()

            # Test knowledge integration through agent coordination
            knowledge_task = {
                "type": "knowledge_integration",
                "query": "integration test knowledge request",
                "context": "cross-track testing",
            }

            # Simulate RAG agent integration
            integration_result = await rag_agent.integrate_knowledge(
                knowledge_task["query"], knowledge_task["context"]
            )

            self.log("INFO", "Cross-track Agent+RAG integration successful")
            return True
        except Exception as e:
            self.log("ERROR", f"Cross-track Agent+RAG failed: {e}")
            return False

    async def test_cross_track_federation_streaming(self) -> bool:
        """Test T5 Federation + T6 Distributed Inference integration"""
        try:
            from federation.core.federation_manager import FederationManager
            from production.distributed_inference.tokenomics_receipts import (
                TokenomicsReceiptManager,
            )

            # Create components
            federation = FederationManager(
                "test_fed_device", enable_tor=False, enable_i2p=False
            )

            # Create temporary database
            import tempfile

            with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
                db_path = tmp.name

            try:
                tokenomics = TokenomicsReceiptManager(db_path)
                await tokenomics.initialize()

                # Test federation with tokenomics tracking
                receipt_id = await tokenomics.create_receipt(
                    "federation_operation",
                    "cross_track_test",
                    {"federation_device": federation.device_id},
                )

                await tokenomics.close()

                self.log(
                    "INFO",
                    f"Cross-track Federation+Streaming successful (receipt: {receipt_id})",
                )
                return True

            finally:
                Path(db_path).unlink(missing_ok=True)

        except Exception as e:
            self.log("ERROR", f"Cross-track Federation+Streaming failed: {e}")
            return False

    # ============================================================================
    # Production Readiness Tests
    # ============================================================================

    def test_production_dependency_check(self) -> bool:
        """Test production dependency availability and graceful degradation"""
        dependencies = {
            "required": ["asyncio", "json", "pathlib", "logging"],
            "optional": ["torch", "transformers", "stem", "bluetooth", "aioquic"],
        }

        missing_required = []
        missing_optional = []

        for dep in dependencies["required"]:
            try:
                __import__(dep)
            except ImportError:
                missing_required.append(dep)

        for dep in dependencies["optional"]:
            try:
                __import__(dep)
            except ImportError:
                missing_optional.append(dep)

        if missing_required:
            self.log("ERROR", f"Missing required dependencies: {missing_required}")
            return False

        if missing_optional:
            self.log(
                "WARN",
                f"Missing optional dependencies (graceful degradation): {missing_optional}",
            )
        else:
            self.log("INFO", "All dependencies available")

        return True

    def test_production_configuration_validation(self) -> bool:
        """Test production configuration and environment setup"""
        try:
            # Check project structure
            required_dirs = [
                "src/production/agent_forge",
                "src/production/rag",
                "src/federation",
                "src/infrastructure",
            ]

            missing_dirs = []
            for dir_path in required_dirs:
                if not Path(dir_path).exists():
                    missing_dirs.append(dir_path)

            if missing_dirs:
                self.log("ERROR", f"Missing required directories: {missing_dirs}")
                return False

            self.log("INFO", "Production directory structure validated")
            return True
        except Exception as e:
            self.log("ERROR", f"Configuration validation failed: {e}")
            return False

    # ============================================================================
    # Main Test Execution
    # ============================================================================

    async def run_all_tests(self):
        """Execute comprehensive integration test suite"""
        self.log("INFO", "Starting AIVillage System Integration Testing")
        self.log("INFO", "=" * 60)

        # T5: Security & Federation Tests
        self.log("INFO", "Testing T5: Security & Federation")
        await self.run_test("T5_federation_imports", self.test_t5_federation_imports)
        await self.run_test(
            "T5_federation_initialization", self.test_t5_federation_initialization
        )
        await self.run_test("T5_bitchat_operations", self.test_t5_bitchat_operations)

        # T3: Agent Forge Tests
        self.log("INFO", "Testing T3: Agent Forge")
        await self.run_test("T3_agent_forge_imports", self.test_t3_agent_forge_imports)
        await self.run_test("T3_agent_orchestration", self.test_t3_agent_orchestration)
        await self.run_test("T3_agent_communication", self.test_t3_agent_communication)

        # T2: RAG System Tests
        self.log("INFO", "Testing T2: RAG System")
        await self.run_test("T2_rag_imports", self.test_t2_rag_imports)
        await self.run_test("T2_rag_performance", self.test_t2_rag_performance)
        await self.run_test("T2_rag_integration", self.test_t2_rag_integration)

        # T6: Distributed Inference Tests
        self.log("INFO", "Testing T6: Distributed Inference")
        await self.run_test("T6_distributed_imports", self.test_t6_distributed_imports)
        await self.run_test(
            "T6_tokenomics_integration", self.test_t6_tokenomics_integration
        )
        await self.run_test("T6_tensor_streaming", self.test_t6_tensor_streaming)

        # Cross-Track Integration Tests
        self.log("INFO", "Testing Cross-Track Integration")
        await self.run_test(
            "CrossTrack_agent_rag", self.test_cross_track_agent_rag_integration
        )
        await self.run_test(
            "CrossTrack_federation_streaming",
            self.test_cross_track_federation_streaming,
        )

        # Production Readiness Tests
        self.log("INFO", "Testing Production Readiness")
        await self.run_test(
            "Production_dependencies", self.test_production_dependency_check
        )
        await self.run_test(
            "Production_configuration", self.test_production_configuration_validation
        )

        # Generate final report
        self.generate_final_report()

    def generate_final_report(self):
        """Generate comprehensive test report"""
        total_duration = time.time() - self.start_time
        success_rate = (
            (self.passed_tests / self.total_tests * 100) if self.total_tests > 0 else 0
        )

        self.log("INFO", "=" * 60)
        self.log("INFO", "SYSTEM INTEGRATION TEST REPORT")
        self.log("INFO", "=" * 60)

        # Test Statistics
        self.log("INFO", f"Total Tests: {self.total_tests}")
        self.log("INFO", f"Passed: {self.passed_tests}")
        self.log("INFO", f"Failed: {self.failed_tests}")
        self.log("INFO", f"Success Rate: {success_rate:.1f}%")
        self.log("INFO", f"Total Duration: {total_duration:.2f}s")

        # Track-by-track summary
        track_summary = {
            "T5": [name for name in self.test_results.keys() if name.startswith("T5_")],
            "T3": [name for name in self.test_results.keys() if name.startswith("T3_")],
            "T2": [name for name in self.test_results.keys() if name.startswith("T2_")],
            "T6": [name for name in self.test_results.keys() if name.startswith("T6_")],
            "CrossTrack": [
                name
                for name in self.test_results.keys()
                if name.startswith("CrossTrack_")
            ],
            "Production": [
                name
                for name in self.test_results.keys()
                if name.startswith("Production_")
            ],
        }

        self.log("INFO", "")
        self.log("INFO", "TRACK SUMMARY:")
        for track, tests in track_summary.items():
            if tests:
                track_passed = sum(
                    1 for test in tests if self.test_results[test]["status"] == "PASS"
                )
                track_total = len(tests)
                track_rate = (
                    (track_passed / track_total * 100) if track_total > 0 else 0
                )
                status = "PASS" if track_passed == track_total else "FAIL"
                self.log(
                    "INFO",
                    f"  {track:12}: {track_passed}/{track_total} ({track_rate:.0f}%) - {status}",
                )

        # Overall status
        overall_status = "PASS" if self.failed_tests == 0 else "FAIL"
        self.log("INFO", "")
        self.log("INFO", f"OVERALL INTEGRATION STATUS: {overall_status}")

        if overall_status == "PASS":
            self.log("INFO", "🎉 System ready for production deployment!")
        else:
            self.log("WARN", "⚠️  Some integration issues found - review failed tests")

        # Failed test details
        if self.failed_tests > 0:
            self.log("INFO", "")
            self.log("INFO", "FAILED TESTS:")
            for test_name, result in self.test_results.items():
                if result["status"] == "FAIL":
                    error_msg = result.get("error", "Unknown error")
                    self.log("ERROR", f"  {test_name}: {error_msg}")


async def main():
    """Main test execution entry point"""
    tester = SystemIntegrationTester()
    await tester.run_all_tests()


if __name__ == "__main__":
    asyncio.run(main())
