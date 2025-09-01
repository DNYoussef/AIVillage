# Bridge for fog marketplace module
try:
    from infrastructure.fog.gateway.scheduler.marketplace import *
except ImportError:
    try:
        from infrastructure.fog.marketplace import *
    except ImportError:
        # Define minimal stubs
        class BidStatus:
            pass
        class BidType:
            pass
        class MarketplaceEngine:
            pass
        class PricingTier:
            pass