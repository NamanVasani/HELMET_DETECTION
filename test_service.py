from pathlib import Path
import plistlib
import unittest

from install_service import service_config


class ServiceTests(unittest.TestCase):
    def test_service_uses_private_runtime_and_restarts_at_login(self):
        runtime = Path("/Users/example/Library/Application Support/Helmet")
        config = plistlib.loads(plistlib.dumps(service_config(runtime)))
        self.assertTrue(config["KeepAlive"])
        self.assertTrue(config["RunAtLoad"])
        self.assertEqual(config["Umask"], 0o077)
        self.assertEqual(config["ProgramArguments"][0], str(runtime / ".venv/bin/python"))
        self.assertNotIn("Desktop", str(config))
        self.assertEqual(config["EnvironmentVariables"]["HELMET_STORAGE"], str(runtime / "storage"))


if __name__ == "__main__":
    unittest.main()
