from token_economy.sovereign_fund import DigitalSovereignWealthFund


def test_barbell_strategy():
    fund = DigitalSovereignWealthFund()
    fund.treasury.total_value = 1000
    fund.implement_barbell_strategy()
    assert fund.investments.conservative["stablecoins"] == 1000 * 0.8 * 0.5
    assert fund.investments.aggressive["ai_startups"] == 1000 * 0.2 * 0.4
