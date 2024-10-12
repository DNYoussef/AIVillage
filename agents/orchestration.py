...
from .king.king_agent import king_agent
from .sage.sage_agent import sage_agent
from .magi.magi_agent import magi_agent

...

def main():
    communication_protocol = StandardCommunicationProtocol()

    agents = [king_agent, sage_agent, magi_agent]
    
    ...
