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
        self.assertEqual(config["ProgramArguments"][-1], str(runtime / "app.py"))
        self.assertNotIn("Desktop", str(config))
        self.assertEqual(config["EnvironmentVariables"]["HELMET_STORAGE"], str(runtime / "storage"))

    def test_lan_mode_requires_login_and_lists_allowed_hosts(self):
        runtime = Path("/Users/example/Library/Application Support/Helmet")
        env = service_config(runtime, ["192.168.1.8:8003", "mac.local:8003"])["EnvironmentVariables"]
        self.assertEqual(env["HELMET_HOST"], "0.0.0.0")
        self.assertEqual(env["HELMET_AUTH"], "1")
        self.assertEqual(env["HELMET_ALLOWED_HOSTS"], "192.168.1.8:8003,mac.local:8003")
        self.assertNotIn("HELMET_HOST", service_config(runtime)["EnvironmentVariables"])


if __name__ == "__main__":
    unittest.main()
