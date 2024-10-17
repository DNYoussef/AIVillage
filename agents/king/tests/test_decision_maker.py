import unittest
from unittest.mock import Mock, patch
from agents.unified_base_agent import UnifiedBaseAgent as Agent
from agents.king.decision_maker import DecisionMaker

class TestDecisionMaker(unittest.TestCase):
    def setUp(self):
        self.agent = Mock(spec=Agent)
        self.decision_maker = DecisionMaker(self.agent)




if __name__ == '__main__':
    unittest.main()
