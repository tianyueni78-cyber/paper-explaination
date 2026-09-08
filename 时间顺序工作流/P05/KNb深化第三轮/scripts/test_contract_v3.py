import json
import unittest
from pathlib import Path

from contract_v3 import validate_protocol


class ContractV3Tests(unittest.TestCase):
    def test_repository_protocol_matches_approved_contract(self):
        path = Path(__file__).resolve().parents[1] / "protocol.json"
        validate_protocol(json.loads(path.read_text(encoding="utf-8")))

    def test_contract_rejects_validation_access_and_qkb(self):
        path = Path(__file__).resolve().parents[1] / "protocol.json"
        protocol = json.loads(path.read_text(encoding="utf-8"))
        protocol["validation_data_opened"] = True
        with self.assertRaises(ValueError):
            validate_protocol(protocol)
        protocol["validation_data_opened"] = False
        protocol["q_kb_enabled"] = True
        with self.assertRaises(ValueError):
            validate_protocol(protocol)


if __name__ == "__main__":
    unittest.main()
