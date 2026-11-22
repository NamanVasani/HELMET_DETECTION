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


class TunnelTests(unittest.TestCase):
    def test_tunnel_mode_trusts_only_quick_tunnel_addresses_with_login(self):
        runtime = Path("/Users/example/Library/Application Support/Helmet")
        env = service_config(runtime, ["192.168.1.9:8003"], tunnel=True)["EnvironmentVariables"]
        self.assertEqual(env["HELMET_AUTH"], "1")
        self.assertEqual(env["HELMET_ALLOWED_HOSTS"], "192.168.1.9:8003,*.trycloudflare.com")
        self.assertNotIn("trycloudflare", str(service_config(runtime)))

    def test_wildcard_host_matching_rejects_look_alikes(self):
        import tempfile
        from dashboard import DashboardServer
        with tempfile.TemporaryDirectory() as temp:
            server = DashboardServer(port=0, storage=Path(temp), allowed_hosts="*.trycloudflare.com,10.0.0.2")
            plain = DashboardServer(port=0, storage=Path(temp))
            try:
                for host in ("calm-river-1.trycloudflare.com", "calm-river-1.trycloudflare.com:443",
                             f"10.0.0.2:{server.port}", f"127.0.0.1:{server.port}"):
                    self.assertTrue(server.host_allowed(host), host)
                for host in ("trycloudflare.com", ".trycloudflare.com", "x.trycloudflare.com.evil.com",
                             "x.trycloudflare.com:8080", "evil.com", "xtrycloudflare.com"):
                    self.assertFalse(server.host_allowed(host), host)
                self.assertFalse(plain.host_allowed("calm-river-1.trycloudflare.com"))
            finally:
                server.close()
                plain.close()

    def test_remote_access_finds_link_and_builds_command(self):
        from remote_access import find_url, tunnel_command
        line = "2026-11-22T10:00:00Z INF |  https://calm-river-1.trycloudflare.com                    |"
        self.assertEqual(find_url(line), "https://calm-river-1.trycloudflare.com")
        self.assertIsNone(find_url("Requesting new quick Tunnel on https://api.trycloudflare.com..."))
        self.assertEqual(tunnel_command("cf")[-3:], ["--url", "https://127.0.0.1:8003", "--no-tls-verify"])
        self.assertEqual(tunnel_command("cf", https=False)[-1], "http://127.0.0.1:8003")
