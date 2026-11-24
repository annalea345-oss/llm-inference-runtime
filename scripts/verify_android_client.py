#!/usr/bin/env python3
"""Opt-in smoke against a running Android emulator and an idle local runtime.

Requires the debug APK already installed. Resets this app's in-memory chat,
temporarily rotates the emulator, and sends real generation requests. No models
are downloaded and no backend is started. See ANDROID_CLIENT.md.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import re
import subprocess
import time
import urllib.request
import xml.etree.ElementTree as ET


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adb", default="adb")
    parser.add_argument("--serial", default="emulator-5554")
    parser.add_argument("--server", default="http://127.0.0.1:8000")
    parser.add_argument("--app-server", default="http://10.0.2.2:8000")
    parser.add_argument("--output", type=Path, default=Path(".experiments/android-client"))
    args = parser.parse_args()
    if not args.serial.startswith("emulator-"):
        parser.error("This smoke changes emulator settings; use an emulator serial.")
    args.output.mkdir(parents=True, exist_ok=True)

    def adb(*parts):
        return subprocess.check_output([args.adb, "-s", args.serial, *parts], timeout=40).decode()

    def health():
        with urllib.request.urlopen(args.server + "/health", timeout=5) as response:
            return json.load(response)

    def idle(h):
        return (h["active_requests"] == h["queued_requests"] == 0
                and h["kv_cache"]["reserved_tokens"] == h["kv_cache"]["used_tokens"] == 0)

    def wait(fn, timeout=45):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            result = fn()
            if result:
                return result
            time.sleep(0.2)
        raise AssertionError("Timed out waiting for smoke condition")

    def nodes():
        raw = adb("exec-out", "uiautomator", "dump", "/dev/tty")
        start = raw.index("<?xml")
        end = raw.index("</hierarchy>") + len("</hierarchy>")
        return list(ET.fromstring(raw[start:end]).iter("node"))

    def texts(ns):
        return "\n".join(n.get("text", "") for n in ns)

    def is_landscape(ns):
        x1, y1, x2, y2 = map(int, re.findall(r"\d+", ns[0].get("bounds")))
        return x2 - x1 > y2 - y1

    def find(ns, label):
        return next(n for n in ns if n.get("content-desc") == label or n.get("text", "").casefold() == label.casefold())

    def tap_node(node):
        x1, y1, x2, y2 = map(int, re.findall(r"\d+", node.get("bounds")))
        adb("shell", "input", "tap", str((x1 + x2) // 2), str((y1 + y2) // 2))

    def tap(label):
        tap_node(find(nodes(), label))

    def fill(label, value):
        n = find(nodes(), label)
        old = n.get("text", "")
        if old == label:
            old = ""  # An empty EditText exposes its hint as text.
        tap_node(n)
        if label == "Output token limit":
            # Numeric fields may ignore MOVE_END on an emulator's hardware keyboard.
            assert find(nodes(), label).get("focused") == "true"
            for _ in range(3):
                adb("shell", "input", "keycombination", "113", "29")  # Ctrl+A
                adb("shell", "input", "keyevent", "KEYCODE_DEL")
                if find(nodes(), label).get("text", "") == "":
                    break
            assert find(nodes(), label).get("text", "") == ""
        else:
            adb("shell", "input", "keyevent", "KEYCODE_MOVE_END")
            if old:
                adb("shell", "input", "keyevent", *(["KEYCODE_DEL"] * len(old)))
        # Test fixtures are ASCII; production Unicode transport is covered by JVM tests.
        adb("shell", "input", "text", value.replace(" ", "%s"))
        if "mInputShown=true" in adb("shell", "dumpsys", "input_method"):
            adb("shell", "input", "keyevent", "KEYCODE_BACK")

    def screen(name):
        data = subprocess.check_output([args.adb, "-s", args.serial, "exec-out", "screencap", "-p"], timeout=15)
        (args.output / (name + ".png")).write_bytes(data)

    def completed():
        ns = nodes()
        ready = "Connected to your Mac" in texts(ns) and any(n.get("resource-id", "").endswith("/assistant_reply") for n in ns)
        if ready:
            assert_bubbles(ns)
        return ns if ready else None

    def assert_bubbles(ns):
        root = list(map(int, re.findall(r"\d+", ns[0].get("bounds"))))
        for item in ns:
            if item.get("resource-id", "").endswith(("/assistant_reply", "/user_message")):
                x1, _, x2, _ = map(int, re.findall(r"\d+", item.get("bounds")))
                assert x2 - x1 > (root[2] - root[0]) // 4, "Message bubble text was collapsed"

    def answer(ns):
        return [n.get("text") for n in ns if n.get("resource-id", "").endswith("/assistant_reply")][-1]

    def settings(address=None, preset=None, exact=None):
        tap("Settings")
        if address is not None:
            fill("Server address", address)
        if preset is not None:
            tap(preset)
        if exact is not None:
            tap("Advanced")
            fill("Output token limit", str(exact))
        ns = nodes()
        tap_node(find(ns, "Save and connect" if any(n.get("text", "").casefold() == "save and connect" for n in ns) else "Save"))

    def reference(messages):
        body = json.dumps({"model": model, "messages": messages, "max_tokens": 64, "temperature": 0}).encode()
        req = urllib.request.Request(args.server + "/v1/chat/completions", data=body, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=90) as response:
            return json.load(response)

    def second_client():
        body = json.dumps({"model": model, "messages": [{"role": "user", "content": "Count from 1 to 500. Include every number."}],
                           "max_tokens": 128, "temperature": 0}).encode()
        req = urllib.request.Request(args.server + "/v1/chat/completions", data=body, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=90) as response:
            return json.load(response)

    initial = health()
    assert idle(initial), "Use a dedicated idle runtime for this smoke."
    old_rotation = adb("shell", "settings", "get", "system", "user_rotation").strip()
    old_accel = adb("shell", "settings", "get", "system", "accelerometer_rotation").strip()
    old_night = adb("shell", "cmd", "uimode", "night").strip().split()[-1]
    old_font = adb("shell", "settings", "get", "system", "font_scale").strip()
    results = {"backend": initial["executor"], "checks": []}
    try:
        adb("shell", "settings", "put", "system", "accelerometer_rotation", "0")
        adb("shell", "settings", "put", "system", "user_rotation", "0")
        adb("shell", "am", "force-stop", "dev.llmruntime.chat")
        adb("shell", "am", "start", "-n", "dev.llmruntime.chat/.MainActivity")
        wait(lambda: "Runtime Chat" in texts(nodes()))
        adb("shell", "cmd", "uimode", "night", "no")
        ns = nodes()
        assert "http://" not in texts(ns) and "64" not in texts(ns), texts(ns)
        screen("welcome")
        results["checks"].append("welcome hides server address and exact token controls")
        if args.app_server != "http://10.0.2.2:8000":
            settings(address=args.app_server)
        else:
            tap("Connect to Mac")
        wait(lambda: "Connected to your Mac" in texts(nodes()))
        with urllib.request.urlopen(args.server + "/v1/models", timeout=5) as response:
            model = json.load(response)["data"][0]["id"]
        tap("Connected to your Mac")
        assert model in texts(nodes())
        tap("Close")
        results["model_alias"] = model
        results["checks"].append("model discovery")
        fill("Message", "My name is Ada. Reply with my name.")
        tap("Send"); completed_first = wait(completed)
        results["first_reply_ui"] = texts(completed_first)
        fill("Message", "What is my name?")
        tap("Send"); followup = wait(completed)
        expected = reference([{"role": "user", "content": "My name is Ada. Reply with my name."},
                              {"role": "assistant", "content": answer(completed_first)},
                              {"role": "user", "content": "What is my name?"}])
        assert answer(followup) == expected["choices"][0]["message"]["content"], texts(followup)
        results["followup_ui"] = texts(followup)
        results["followup_direct_reference"] = expected
        results["checks"].append("streamed follow-up matches direct API with exact history")
        screen("portrait")
        adb("shell", "settings", "put", "system", "user_rotation", "1")
        rotated = wait(lambda: (n if is_landscape(n := nodes()) and any(
            item.get("resource-id", "").endswith("/assistant_reply") and item.get("text") == expected["choices"][0]["message"]["content"]
            for item in n) else None))
        assert find(rotated, "Send").get("bounds") != "[0,0][0,0]"
        assert_bubbles(rotated)
        screen("landscape")
        results["checks"].append("rotation preserves completed history and controls")
        adb("shell", "settings", "put", "system", "user_rotation", "0")
        wait(lambda: not is_landscape(nodes()))
        adb("shell", "cmd", "uimode", "night", "yes")
        wait(lambda: "Connected to your Mac" in texts(nodes()))
        assert answer(nodes()) == expected["choices"][0]["message"]["content"]
        screen("dark")
        results["checks"].append("dark theme retains conversation")
        adb("shell", "settings", "put", "system", "font_scale", "1.3")
        wait(lambda: "Connected to your Mac" in texts(nodes()))
        screen("large-text")
        results["checks"].append("large text layout inspected")
        adb("shell", "settings", "put", "system", "font_scale", old_font)
        adb("shell", "cmd", "uimode", "night", "no")
        settings(preset="Long")
        results["checks"].append("Long reply preset saved through Settings")
        fill("Message", "Count from 1 to 500. Include every number.")
        before = nodes()
        tap_node(find(before, "Send"))
        wait(lambda: health()["active_requests"] > 0)
        with ThreadPoolExecutor(max_workers=1) as pool:
            other = pool.submit(second_client)
            results["concurrent_health"] = wait(lambda: (h if (h := health())["active_requests"] == 2 else None))
            results["checks"].append("two simultaneous clients admitted")
            tap_node(find(before, "Send"))
            stopped = wait(lambda: (n if "Reply stopped" in texts(n := nodes()) else None))
            results["stopped_ui"] = texts(stopped)
            result = other.result(timeout=90)
            assert result["choices"][0]["finish_reason"] in ("stop", "length")
            results["second_client_usage"] = result["usage"]
        wait(lambda: idle(health()))
        results["checks"].append("Stop and peer completion leave zero request/KV state")
        screen("stopped")
        tap("New chat")
        assert not any(n.get("resource-id", "").endswith("/user_message") for n in nodes())
        results["checks"].append("new chat clears displayed conversation")
        settings(address="http://10.0.2.2:1")
        wait(lambda: "Couldn’t connect." in texts(nodes()))
        results["checks"].append("offline server shows connection error")
        settings(address=args.app_server)
        wait(lambda: "Connected to your Mac" in texts(nodes()))
        results["final_health"] = health()
        results["checks"].append("reconnect succeeds")
    finally:
        adb("shell", "cmd", "uimode", "night", old_night)
        adb("shell", "settings", "put", "system", "font_scale", old_font)
        adb("shell", "settings", "put", "system", "user_rotation", old_rotation)
        adb("shell", "settings", "put", "system", "accelerometer_rotation", old_accel)
        (args.output / "smoke.json").write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"checks": results["checks"], "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
