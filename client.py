# client.py
import argparse
import asyncio
import json
import uuid

HOST = "127.0.0.1"
PORT = 5000

async def send_command(command: str, params):
    reader, writer = await asyncio.open_connection(HOST, PORT)
    mid = uuid.uuid4().hex[:8]
    message = {"id": mid, "command": command, "params": params}
    writer.write((json.dumps(message) + "\n").encode("utf-8"))
    await writer.drain()
    resp_line = await reader.readline()
    if not resp_line:
        print("No response")
    else:
        resp = json.loads(resp_line.decode("utf-8"))
        print(json.dumps(resp, indent=2))
    writer.close()
    await writer.wait_closed()

def parse_value(v):
    # simple parser from CLI strings: JSON if possible else raw string
    import json
    try:
        return json.loads(v)
    except Exception:
        return v

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command")
    # Allow multiple -p occurrences, each accepting one or more key=val entries
    parser.add_argument(
        "--param", "-p", action="append", nargs="+", dest="params",
        help="params as key=val (val JSON if possible); repeat -p or provide multiple key=val after one -p",
        default=[],
    )
    args = parser.parse_args()
    params = {}
    # Flatten list of lists from action=append + nargs='+'
    kv_items = [item for group in (args.params or []) for item in group]
    if kv_items:
        for kv in kv_items:
            if "=" not in kv:
                continue
            k, v = kv.split("=", 1)
            params[k] = parse_value(v)
    asyncio.run(send_command(args.command, params))
