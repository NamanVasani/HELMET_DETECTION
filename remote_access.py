"""Open the Helmet dashboard from anywhere (mobile data, another Wi-Fi) with a free Cloudflare quick tunnel.

No account and no router changes: cloudflared makes an outbound connection and prints a random
https://<words>.trycloudflare.com address with a real certificate, so phone cameras work without warnings.
The address changes every run and stops working when you press Ctrl+C.

Before the first run:
  1. Change the admin password (Settings -> Users), because the link is reachable from the internet.
  2. brew install cloudflared
  3. python install_service.py --lan --tunnel --replace-pid "$(lsof -tnP -iTCP:8003 -sTCP:LISTEN)"
Then: python remote_access.py
"""
import argparse
import re
import shutil
import subprocess
import sys

PORT = 8003  # fixed for Helmet (see .cursor/rules/local-ports.mdc)
URL_RE = re.compile(r"https://[a-z0-9-]+\.trycloudflare\.com")


def tunnel_command(cloudflared, https=True):
    origin = f"{'https' if https else 'http'}://127.0.0.1:{PORT}"
    # The local service uses a self-signed certificate; the tunnel itself is verified by Cloudflare.
    return [cloudflared, "tunnel", "--no-autoupdate", "--url", origin] + (["--no-tls-verify"] if https else [])


def find_url(line):
    match = URL_RE.search(line)
    return match.group(0) if match and "api.trycloudflare.com" not in match.group(0) else None


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--http", action="store_true", help="The service runs without --lan (plain http locally)")
    args = parser.parse_args(argv)
    cloudflared = shutil.which("cloudflared")
    if not cloudflared:
        print("cloudflared is not installed. Install it (free) with:  brew install cloudflared")
        return 1
    process = subprocess.Popen(tunnel_command(cloudflared, https=not args.http), stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, text=True)
    shown = False
    try:
        for line in process.stdout:
            url = find_url(line)
            if url and not shown:
                shown = True
                print(f"\nHelmet is reachable from anywhere at:\n\n    {url}\n")
                print("Open it on your phone (mobile data works too) and sign in. Press Ctrl+C to close the link.")
                print("If it says 'Use this dashboard from its configured address', re-run install_service.py "
                      "with --tunnel.\n")
            elif "ERR" in line and not shown:
                print(line.rstrip())
    except KeyboardInterrupt:
        pass
    finally:
        process.terminate()
        try:
            process.wait(5)
        except subprocess.TimeoutExpired:
            process.kill()
    print("Tunnel closed; the public link no longer works.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
