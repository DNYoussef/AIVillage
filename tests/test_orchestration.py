#!/usr/bin/env python3
"""
Test the multi-model orchestration system.

This script tests the OpenRouter integration with the Agent Forge curriculum learning system.
"""

import asyncio
import json
import logging
from pathlib import Path
from typing import Dict, Any

from agent_forge.orchestration import (
    OpenRouterClient,
    TaskRouter,
    TaskType,
    MODEL_ROUTING_CONFIG
)
from agent_forge.orchestration.task_router import TaskContext
from agent_forge.orchestration.curriculum_integration import (
    MultiModelOrchestrator,
    EnhancedFrontierQuestionGenerator
)
from agent_forge.training.magi_specialization import MagiConfig


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class OrchestrationTester:
    """Test harness for multi-model orchestration."""
    
    def __init__(self):
        self.results = {}
        self.client = None
        self.router = None
        self.orchestrator = None
    
    async def test_openrouter_client(self):
        """Test basic OpenRouter client functionality."""
        logger.info("Testing OpenRouter client...")
        
        try:
            self.client = OpenRouterClient()
            
            # Test simple completion
            response = await self.client.complete(
                task_type=TaskType.EVALUATION_GRADING,
                messages=[{"role": "user", "content": "What is 2+2? Reply with just the number."}],
                max_tokens=10,
                temperature=0.1
            )
            
            logger.info(f"Response: {response.content}")
            logger.info(f"Model used: {response.model_used}")
            logger.info(f"Cost: ${response.cost:.4f}")
            logger.info(f"Latency: {response.latency:.2f}s")
            
            self.results['client_test'] = {
                'success': True,
                'response': response.content,
                'model': response.model_used,
                'cost': response.cost
            }
            
            return True
            
        except Exception as e:
            logger.error(f"Client test failed: {e}")
            self.results['client_test'] = {
                'success': False,
                'error': str(e)
            }
            return False
    
    async def test_task_routing(self):
        """Test task classification and routing."""
        logger.info("Testing task routing...")
        
        try:
            self.router = TaskRouter(self.client)
            
            test_cases = [
                {
                    'prompt': "Generate a Python programming challenge about recursion",
                    'expected_type': TaskType.CODE_GENERATION,
                    'context': TaskContext(
                        difficulty_level=6,
                        domain="python_programming",
                        expected_length="medium",
                        requires_reasoning=True,
                        requires_creativity=True
                    )
                },
                {
                    'prompt': "Evaluate this solution and provide a grade",
                    'expected_type': TaskType.EVALUATION_GRADING,
                    'context': TaskContext(
                        difficulty_level=3,
                        domain="evaluation",
                        expected_length="short",
                        requires_reasoning=True,
                        requires_creativity=False,
                        cost_sensitive=True
                    )
                },
                {
                    'prompt': "Prove that the sum of angles in a triangle is 180 degrees",
                    'expected_type': TaskType.MATHEMATICAL_REASONING,
                    'context': TaskContext(
                        difficulty_level=7,
                        domain="mathematical_proofs",
                        expected_length="long",
                        requires_reasoning=True,
                        requires_creativity=False
                    )
                }
            ]
            
            routing_results = []
            
            for test in test_cases:
                # Test classification
                classified_type = self.router.classify_task(test['prompt'], test['context'])
                logger.info(f"Classified '{test['prompt'][:50]}...' as {classified_type.value}")
                
                # Test model selection
                selected_model = self.router.select_model_for_task(classified_type, test['context'])
                logger.info(f"Selected model: {selected_model}")
                
                # Test actual routing (small request)
                response = await self.router.route_task(
                    test['prompt'],
                    test['context'],
                    max_tokens=100
                )
                
                routing_results.append({
                    'prompt': test['prompt'][:50],
                    'expected': test['expected_type'].value,
                    'classified': classified_type.value,
                    'correct': classified_type == test['expected_type'],
                    'model': selected_model,
                    'response_preview': response.content[:100] + "..."
                })
            
            self.results['routing_test'] = {
                'success': True,
                'results': routing_results,
                'accuracy': sum(r['correct'] for r in routing_results) / len(routing_results)
            }
            
            return True
            
        except Exception as e:
            logger.error(f"Routing test failed: {e}")
            self.results['routing_test'] = {
                'success': False,
                'error': str(e)
            }
            return False
    
    async def test_problem_generation_with_variations(self):
        """Test problem generation with variations."""
        logger.info("Testing problem generation with variations...")
        
        try:
            result = await self.router.generate_problem_with_variations(
                domain="algorithm_design",
                difficulty=6,
                num_variations=2
            )
            
            logger.info(f"Original problem: {result['original'][:200]}...")
            logger.info(f"Generated {len(result['variations'])} variations")
            logger.info(f"Total cost: ${result['total_cost']:.4f}")
            
            self.results['variation_test'] = {
                'success': True,
                'num_variations': len(result['variations']),
                'total_cost': result['total_cost'],
                'models_used': result['models_used']
            }
            
            return True
            
        except Exception as e:
            logger.error(f"Variation test failed: {e}")
            self.results['variation_test'] = {
                'success': False,
                'error': str(e)
            }
            return False
    
    async def test_curriculum_integration(self):
        """Test integration with curriculum learning system."""
        logger.info("Testing curriculum integration...")
        
        try:
            # Create test configuration
            config = MagiConfig(
                curriculum_levels=2,
                questions_per_level=5,
                total_questions=10
            )
            
            # Initialize orchestrator
            self.orchestrator = MultiModelOrchestrator(config, enable_openrouter=True)
            
            # Test enhanced question generation
            generator = self.orchestrator.question_generator
            
            # Generate a single question
            question = generator._generate_single_question("python_programming", 5)
            
            logger.info(f"Generated question: {question.text[:200]}...")
            logger.info(f"Domain: {question.domain}, Difficulty: {question.difficulty}")
            
            # Test evaluation
            test_answer = "This is a test answer to the programming question."
            eval_result = await self.orchestrator.evaluate_answer_with_explanation(
                question,
                test_answer
            )
            
            logger.info(f"Evaluation result: {eval_result}")
            
            self.results['integration_test'] = {
                'success': True,
                'question_generated': True,
                'evaluation_completed': True,
                'eval_result': str(eval_result)[:200]
            }
            
            return True
            
        except Exception as e:
            logger.error(f"Integration test failed: {e}")
            self.results['integration_test'] = {
                'success': False,
                'error': str(e)
            }
            return False
    
    async def test_cost_tracking(self):
        """Test cost tracking and metrics."""
        logger.info("Testing cost tracking...")
        
        try:
            if self.client:
                metrics = self.client.get_metrics_summary()
                
                logger.info("Cost tracking metrics:")
                logger.info(f"Total cost: ${metrics['total_cost']:.4f}")
                logger.info(f"Cost by task: {json.dumps(metrics['cost_by_task'], indent=2)}")
                
                self.results['cost_tracking'] = {
                    'success': True,
                    'metrics': metrics
                }
                
                return True
            else:
                logger.warning("No client available for cost tracking test")
                return False
                
        except Exception as e:
            logger.error(f"Cost tracking test failed: {e}")
            self.results['cost_tracking'] = {
                'success': False,
                'error': str(e)
            }
            return False
    
    async def run_all_tests(self):
        """Run all orchestration tests."""
        logger.info("Starting orchestration tests...")
        
        # Run tests in sequence
        await self.test_openrouter_client()
        
        if self.results.get('client_test', {}).get('success'):
            await self.test_task_routing()
            await self.test_problem_generation_with_variations()
            await self.test_curriculum_integration()
            await self.test_cost_tracking()
        
        # Clean up
        if self.orchestrator:
            await self.orchestrator.close()
        elif self.client:
            await self.client.close()
        
        # Summary
        logger.info("\n" + "="*50)
        logger.info("TEST SUMMARY")
        logger.info("="*50)
        
        for test_name, result in self.results.items():
            status = "✓ PASSED" if result.get('success') else "✗ FAILED"
            logger.info(f"{test_name}: {status}")
            
            if not result.get('success'):
                logger.error(f"  Error: {result.get('error')}")
        
        # Save results
        with open('orchestration_test_results.json', 'w') as f:
            json.dump(self.results, f, indent=2)
        
        logger.info(f"\nResults saved to orchestration_test_results.json")
        
        # Return overall success
        return all(r.get('success', False) for r in self.results.values())


async def main():
    """Main test runner."""
    tester = OrchestrationTester()
    success = await tester.run_all_tests()
    
    if success:
        logger.info("\n✅ All tests passed!")
    else:
        logger.error("\n❌ Some tests failed. Check the logs for details.")
    
    return success


if __name__ == "__main__":
    # Run the tests
    success = asyncio.run(main())
    exit(0 if success else 1)