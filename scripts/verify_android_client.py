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
        adb("shell", "input", "keyevent", "KEYCODE_MOVE_END")
        if old:
            adb("shell", "input", "keyevent", *(["KEYCODE_DEL"] * len(old)))
        # Test fixtures are ASCII; production Unicode transport is covered by JVM tests.
        adb("shell", "input", "text", value.replace(" ", "%s"))
        adb("shell", "input", "keyevent", "KEYCODE_BACK")

    def screen(name):
        data = subprocess.check_output([args.adb, "-s", args.serial, "exec-out", "screencap", "-p"], timeout=15)
        (args.output / (name + ".png")).write_bytes(data)

    def completed():
        ns = nodes()
        return ns if "output tokens ·" in texts(ns) else None

    def answer(ns):
        transcript = next(n.get("text") for n in ns if n.get("text", "").startswith("\nYOU\n"))
        return transcript.rsplit("\nASSISTANT\n", 1)[-1]

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
    results = {"backend": initial["executor"], "checks": []}
    try:
        adb("shell", "settings", "put", "system", "accelerometer_rotation", "0")
        adb("shell", "settings", "put", "system", "user_rotation", "0")
        adb("shell", "am", "force-stop", "dev.llmruntime.chat")
        adb("shell", "am", "start", "-n", "dev.llmruntime.chat/.MainActivity")
        wait(lambda: "Runtime Chat" in texts(nodes()))
        if args.app_server != "http://10.0.2.2:8000":
            fill("Server address", args.app_server)
        tap("Connect")
        ns = wait(lambda: (n if "Connected ·" in texts(n := nodes()) else None))
        model = next(n.get("text").split("Connected · ", 1)[1] for n in ns if n.get("text", "").startswith("Connected · "))
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
        rotated = wait(lambda: (n if is_landscape(n := nodes()) and "What is my name?" in texts(n) else None))
        assert find(rotated, "Send").get("enabled") == "true"
        assert find(rotated, "Stop").get("bounds") != "[0,0][0,0]"
        screen("landscape")
        results["checks"].append("rotation preserves completed history and controls")
        adb("shell", "settings", "put", "system", "user_rotation", "0")
        wait(lambda: not is_landscape(nodes()))
        fill("Output token limit", "512")
        fill("Message", "Count from 1 to 500. Include every number.")
        before = nodes()
        tap_node(find(before, "Send"))
        wait(lambda: health()["active_requests"] > 0)
        with ThreadPoolExecutor(max_workers=1) as pool:
            other = pool.submit(second_client)
            results["concurrent_health"] = wait(lambda: (h if (h := health())["active_requests"] == 2 else None))
            results["checks"].append("two simultaneous clients admitted")
            tap_node(find(before, "Stop"))
            stopped = wait(lambda: (n if "Stopped ·" in texts(n := nodes()) else None))
            results["stopped_ui"] = texts(stopped)
            result = other.result(timeout=90)
            assert result["choices"][0]["finish_reason"] in ("stop", "length")
            results["second_client_usage"] = result["usage"]
        wait(lambda: idle(health()))
        results["checks"].append("Stop and peer completion leave zero request/KV state")
        screen("stopped")
        tap("New chat")
        assert "YOU\n" not in texts(nodes())
        results["checks"].append("new chat clears displayed conversation")
        fill("Server address", "http://10.0.2.2:1")
        tap("Connect")
        wait(lambda: "Could not connect:" in texts(nodes()))
        results["checks"].append("offline server shows connection error")
        fill("Server address", args.app_server)
        tap("Connect")
        wait(lambda: "Connected ·" in texts(nodes()))
        results["final_health"] = health()
        results["checks"].append("reconnect succeeds")
    finally:
        adb("shell", "settings", "put", "system", "user_rotation", old_rotation)
        adb("shell", "settings", "put", "system", "accelerometer_rotation", old_accel)
        (args.output / "smoke.json").write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"checks": results["checks"], "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
