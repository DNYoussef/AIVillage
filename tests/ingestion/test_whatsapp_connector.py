from pathlib import Path

from ingestion.connectors.whatsapp import WhatsAppConnector

DATA_DIR = Path(__file__).parent.parent / "data"


def test_parse_txt_export():
    path = DATA_DIR / "whatsapp_chat.txt"
    conn = WhatsAppConnector()
    count = conn.load_export(path)
    assert count == 3
    msgs = conn.get_messages()
    assert msgs[0]["sender"] == "Alice"
    assert conn.get_message_count() == 3


def test_parse_zip_export(tmp_path):
    txt_path = DATA_DIR / "whatsapp_chat.txt"
    zip_path = tmp_path / "export.zip"
    import zipfile

    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.write(txt_path, arcname="whatsapp.txt")

    conn = WhatsAppConnector()
    conn.load_export(zip_path)
    assert conn.get_message_count() == 3
