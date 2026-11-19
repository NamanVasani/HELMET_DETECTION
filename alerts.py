"""Background alerts for new violation events: JSON webhook (Slack/Teams/WhatsApp gateways) and SMTP email."""

import base64
from email.message import EmailMessage
import json
import logging
from pathlib import Path
import queue
import re
import smtplib
import ssl
import threading
import time
from urllib.parse import urlencode, urlparse
from urllib.request import Request, urlopen

from event_store import get_setting, record_audit, set_setting

DEFAULTS = {
    "enabled": False,
    "categories": ["no_helmet"],
    "min_interval_seconds": 60,
    "dashboard_url": "http://127.0.0.1:8003",
    "webhook_url": "",
    "webhook_token": "",
    "email_enabled": False,
    "smtp_host": "",
    "smtp_port": 587,
    "smtp_user": "",
    "smtp_password": "",
    "smtp_from": "",
    "email_to": "",
    "whatsapp_enabled": False,
    "twilio_account_sid": "",
    "twilio_auth_token": "",
    "whatsapp_from": "",
    "whatsapp_to": "",
}
SECRET_FIELDS = ("webhook_token", "smtp_password", "twilio_auth_token")
TWILIO_API = "https://api.twilio.com/2010-04-01/Accounts/{sid}/Messages.json"
PHONE_RE = re.compile(r"\+[1-9]\d{7,14}\Z")


def load_settings(root):
    return {**DEFAULTS, **(get_setting(root, "alerts", {}) or {})}


def public_settings(settings):
    """Settings safe to send to the browser: secrets replaced by a set/unset flag."""
    shown = dict(settings)
    for name in SECRET_FIELDS:
        shown[name] = ""
        shown[name + "_set"] = bool(settings.get(name))
    return shown


def save_settings(root, incoming):
    current = load_settings(root)
    updated = dict(current)
    for key, default in DEFAULTS.items():
        if key not in incoming:
            continue
        value = incoming[key]
        if key in SECRET_FIELDS and value == "":
            continue  # blank means "keep the stored secret"
        if isinstance(default, bool):
            value = bool(value)
        elif isinstance(default, int):
            value = int(value)
            if key == "min_interval_seconds" and not 0 <= value <= 86400:
                raise ValueError("Alert interval must be 0-86400 seconds")
            if key == "smtp_port" and not 1 <= value <= 65535:
                raise ValueError("Invalid SMTP port")
        elif isinstance(default, list):
            value = [item for item in value if item in {"helmet", "no_helmet"}]
        else:
            value = str(value).strip()[:500]
        updated[key] = value
    for key in ("webhook_url", "dashboard_url"):
        if updated[key] and urlparse(updated[key]).scheme not in {"http", "https"}:
            raise ValueError(f"{key.replace('_', ' ').capitalize()} must start with http:// or https://")
    if updated["email_enabled"] and not (updated["smtp_host"] and updated["email_to"] and updated["smtp_from"]):
        raise ValueError("Email alerts need an SMTP host, a from address, and at least one recipient")
    for key in ("whatsapp_from", "whatsapp_to"):
        updated[key] = ",".join(number.strip().removeprefix("whatsapp:").replace(" ", "")
                                for number in updated[key].split(",") if number.strip())
    if updated["whatsapp_enabled"]:
        numbers = [updated["whatsapp_from"]] + updated["whatsapp_to"].split(",")
        if not re.fullmatch(r"AC[0-9a-fA-F]{32}", updated["twilio_account_sid"]) or not updated["twilio_auth_token"]:
            raise ValueError("WhatsApp alerts need your Twilio Account SID (starts with AC) and Auth Token")
        if not updated["whatsapp_to"] or not all(PHONE_RE.fullmatch(number) for number in numbers):
            raise ValueError("WhatsApp numbers must be in international format, e.g. +919876543210")
    for name in SECRET_FIELDS:
        if incoming.get(name + "_clear"):
            updated[name] = ""
    set_setting(root, "alerts", updated)
    return updated


def message_for(event, settings):
    detection = event.get("event_detection") or {}
    source = event.get("source") or {}
    label = "No helmet" if event.get("category") == "no_helmet" else "Helmet"
    camera = source.get("camera") or source.get("filename") or source.get("type") or "camera"
    link = f"{settings['dashboard_url'].rstrip('/')}/?date={event['date']}&event={event['id']}"
    plate = event.get("plate_text") or "not read"
    text = (f"{label} detected on {camera} at {event['timestamp']} "
            f"(score {detection.get('confidence', 0):.2f}, plate {plate}). Needs human review: {link}")
    return text, link


def send_webhook(settings, event, text, link):
    body = json.dumps({"text": text, "event": {
        "id": event["id"], "date": event["date"], "timestamp": event["timestamp"],
        "category": event.get("category"), "plate_text": event.get("plate_text"),
        "camera": (event.get("source") or {}).get("camera"), "link": link,
        "confidence": (event.get("event_detection") or {}).get("confidence")}}).encode()
    headers = {"Content-Type": "application/json", "User-Agent": "helmet-alerts/1.0"}
    if settings.get("webhook_token"):
        headers["Authorization"] = f"Bearer {settings['webhook_token']}"
    with urlopen(Request(settings["webhook_url"], data=body, headers=headers, method="POST"), timeout=10) as response:
        if response.status >= 300:
            raise OSError(f"Webhook returned HTTP {response.status}")


def send_email(settings, event, text, attachment=None):
    message = EmailMessage()
    message["Subject"] = f"Helmet alert: {event.get('category', 'event').replace('_', ' ')} at {event['timestamp']}"
    message["From"] = settings["smtp_from"]
    message["To"] = [address.strip() for address in settings["email_to"].split(",") if address.strip()]
    message.set_content(text + "\n\nThis is an automated suggestion from a detection model, not proof of a violation.")
    if attachment:
        message.add_attachment(attachment, maintype="image", subtype="jpeg", filename="person.jpg")
    port = int(settings["smtp_port"])
    context = ssl.create_default_context()
    if port == 465:
        server = smtplib.SMTP_SSL(settings["smtp_host"], port, timeout=15, context=context)
    else:
        server = smtplib.SMTP(settings["smtp_host"], port, timeout=15)
        server.starttls(context=context)
    with server:
        if settings.get("smtp_user"):
            server.login(settings["smtp_user"], settings["smtp_password"])
        server.send_message(message)


def send_whatsapp(settings, text):
    """Send the alert text to each WhatsApp number via Twilio's Messages API.

    No photo: Twilio can only attach media from a public URL, and the dashboard is private.
    """
    url = TWILIO_API.format(sid=settings["twilio_account_sid"])
    token = base64.b64encode(f"{settings['twilio_account_sid']}:{settings['twilio_auth_token']}".encode()).decode()
    headers = {"Authorization": f"Basic {token}", "Content-Type": "application/x-www-form-urlencoded",
               "User-Agent": "helmet-alerts/1.0"}
    for number in filter(None, settings["whatsapp_to"].split(",")):
        body = urlencode({"From": f"whatsapp:{settings['whatsapp_from']}", "To": f"whatsapp:{number}",
                          "Body": text[:1500]}).encode()
        with urlopen(Request(url, data=body, headers=headers, method="POST"), timeout=15) as response:
            if response.status >= 300:
                raise OSError(f"Twilio returned HTTP {response.status}")


class AlertDispatcher:
    """Queue events and deliver alerts on a background thread, never blocking detection."""

    def __init__(self, root, clock=time.monotonic):
        self.root = root
        self.clock = clock
        self.queue = queue.Queue(maxsize=100)
        self.last_sent = None
        self.last_error = ""
        self.sent = 0
        self.thread = threading.Thread(target=self._run, daemon=True, name="helmet-alerts")
        self.thread.start()

    def notify(self, event, image_path=None):
        try:
            self.queue.put_nowait((event, image_path))
        except queue.Full:
            logging.warning("Alert queue full; dropping alert for %s", event.get("id"))

    def deliver(self, event, image_path=None, force=False):
        settings = load_settings(self.root)
        if not force:
            if not settings["enabled"] or event.get("category") not in settings["categories"]:
                return "disabled"
            now = self.clock()
            if self.last_sent is not None and now - self.last_sent < settings["min_interval_seconds"]:
                return "rate_limited"
        text, link = message_for(event, settings)
        channels = []
        if settings["webhook_url"]:
            send_webhook(settings, event, text, link)
            channels.append("webhook")
        if settings["email_enabled"]:
            attachment = None
            if image_path:
                try:
                    attachment = Path(image_path).read_bytes()
                except OSError:
                    attachment = None
            send_email(settings, event, text, attachment)
            channels.append("email")
        if settings["whatsapp_enabled"]:
            send_whatsapp(settings, text)
            channels.append("whatsapp")
        if not channels:
            raise ValueError("No alert channel is configured")
        self.last_sent = self.clock()
        self.sent += 1
        return "sent:" + ",".join(channels)

    def _run(self):
        while True:
            event, image_path = self.queue.get()
            try:
                result = self.deliver(event, image_path)
                if result.startswith("sent"):
                    self.last_error = ""
                    record_audit(self.root, "system", "alert_sent", f"{event['date']}/{event['id']}", {"result": result})
            except Exception as exc:  # network/SMTP failures must never stop detection
                self.last_error = f"{type(exc).__name__}: {exc}"[:300]
                logging.warning("Alert delivery failed: %s", self.last_error)
                record_audit(self.root, "system", "alert_failed", f"{event.get('date')}/{event.get('id')}",
                             {"error": self.last_error})
