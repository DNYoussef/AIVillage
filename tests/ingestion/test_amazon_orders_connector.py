from pathlib import Path

from ingestion.connectors.amazon_orders import AmazonOrdersConnector

DATA_DIR = Path(__file__).parent.parent / 'data'


def test_parse_csv_export():
    path = DATA_DIR / 'amazon_orders.csv'
    conn = AmazonOrdersConnector()
    count = conn.load_export(path)
    assert count == 2
    orders = conn.get_orders()
    assert orders[0]['Title'] == 'USB Cable'
    assert conn.get_order_count() == 2
