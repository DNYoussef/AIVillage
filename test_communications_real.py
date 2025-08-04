#!/usr/bin/env python3
"""
REAL TEST of communications protocol - no assumptions
"""
import asyncio
import sys
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

async def test_communications_protocol():
    """Test if the communications protocol actually works"""
    print("TESTING COMMUNICATIONS PROTOCOL...")
    
    try:
        # Try to import
        from communications.protocol import CommunicationsProtocol
        print("[OK] Import successful")
        
        # Try to create instance
        protocol = CommunicationsProtocol(agent_id="test_agent", port=8889)
        print("[OK] Instance creation successful")
        
        # Try to start server
        print("Starting server...")
        await protocol.start_server()
        print("[OK] Server started")
        
        # Test connection (to self)
        print("Testing self-connection...")
        success = await protocol.connect("ws://localhost:8889", "test_target")
        print(f"Connection result: {success}")
        
        # Test message sending
        if success:
            print("Testing message sending...")
            message = {"type": "test", "content": "Hello World", "timestamp": time.time()}
            send_result = await protocol.send_message("test_target", message)
            print(f"Send result: {send_result}")
        
        # Stop server
        await protocol.stop_server()
        print("[OK] Server stopped")
        
        return True
        
    except ImportError as e:
        print(f"[FAIL] IMPORT FAILED: {e}")
        return False
    except Exception as e:
        print(f"[FAIL] TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_whatsapp_connector():
    """Test if WhatsApp connector actually works"""
    print("\nTESTING WHATSAPP CONNECTOR...")
    
    try:
        from ingestion.connectors.whatsapp import WhatsAppConnector
        print("[OK] Import successful")
        
        connector = WhatsAppConnector()
        print("[OK] Instance creation successful")
        
        # Test getting message count
        count = connector.get_message_count()
        print(f"Message count: {count}")
        
        # Test getting messages
        messages = connector.get_messages(5)
        print(f"Retrieved {len(messages)} messages")
        
        if messages:
            print(f"First message: {messages[0]}")
        
        return len(messages) > 0
        
    except ImportError as e:
        print(f"[FAIL] IMPORT FAILED: {e}")
        return False
    except Exception as e:
        print(f"[FAIL] TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_amazon_connector():
    """Test if Amazon connector actually works"""
    print("\nTESTING AMAZON CONNECTOR...")
    
    try:
        from ingestion.connectors.amazon_orders import AmazonOrdersConnector
        print("[OK] Import successful")
        
        connector = AmazonOrdersConnector()
        print("[OK] Instance creation successful")
        
        # Test getting order count
        count = connector.get_order_count()
        print(f"Order count: {count}")
        
        # Test getting orders
        orders = connector.get_orders(5)
        print(f"Retrieved {len(orders)} orders")
        
        if orders:
            print(f"First order: {orders[0]}")
        
        return len(orders) > 0
        
    except ImportError as e:
        print(f"[FAIL] IMPORT FAILED: {e}")
        return False
    except Exception as e:
        print(f"[FAIL] TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """Run all real tests"""
    print("REAL TESTING - NO ASSUMPTIONS")
    print("=" * 50)
    
    results = []
    
    # Test communications
    comm_result = await test_communications_protocol()
    results.append(("Communications", comm_result))
    
    # Test WhatsApp
    whatsapp_result = await test_whatsapp_connector()
    results.append(("WhatsApp", whatsapp_result))
    
    # Test Amazon
    amazon_result = await test_amazon_connector()
    results.append(("Amazon", amazon_result))
    
    # Summary
    print("\n" + "=" * 50)
    print("REAL TEST RESULTS:")
    for name, result in results:
        status = "[PASS]" if result else "[FAIL]"
        print(f"{name}: {status}")
    
    all_passed = all(result for _, result in results)
    print(f"\nOVERALL: {'ALL PASSED' if all_passed else 'SOME FAILED'}")
    
    return all_passed

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)