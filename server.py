# server.py
import asyncio
import json
import traceback
from ml_engine import (
    list_datasets, train_model, evaluate_model, predict,
    explain_prediction, list_models, export_model
)

HOST = "127.0.0.1"
PORT = 5000

COMMANDS = {
    "list_datasets": list_datasets,
    "train_model": train_model,
    "evaluate_model": evaluate_model,
    "predict": predict,
    "explain_prediction": explain_prediction,
    "list_models": list_models,
    "export_model": export_model
}

async def handle_client(reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
    addr = writer.get_extra_info('peername')
    print(f"Client connected: {addr}")
    try:
        while not reader.at_eof():
            line = await reader.readline()
            if not line:
                break
            try:
                message = json.loads(line.decode('utf-8').strip())
            except Exception:
                resp = {"id": None, "status": "error", "result": "invalid json"}
                writer.write((json.dumps(resp) + "\n").encode("utf-8"))
                await writer.drain()
                continue

            mid = message.get("id")
            cmd = message.get("command")
            params = message.get("params", {})

            print(f"Received command: {cmd} id={mid} params={params}")

            if cmd not in COMMANDS:
                resp = {"id": mid, "status": "error", "result": f"unknown command {cmd}"}
                writer.write((json.dumps(resp) + "\n").encode("utf-8"))
                await writer.drain()
                continue

            try:
                # Call target function
                fn = COMMANDS[cmd]
                # allow both dict params or named arguments
                if isinstance(params, dict):
                    result = fn(**params)
                else:
                    result = fn(params)
                resp = {"id": mid, "status": "ok", "result": result}
            except Exception as e:
                tb = traceback.format_exc()
                resp = {"id": mid, "status": "error", "result": str(e), "trace": tb}
            writer.write((json.dumps(resp) + "\n").encode("utf-8"))
            await writer.drain()
    except asyncio.CancelledError:
        pass
    finally:
        writer.close()
        await writer.wait_closed()
        print(f"Client disconnected: {addr}")

async def main():
    server = await asyncio.start_server(handle_client, HOST, PORT)
    addrs = ", ".join(str(sock.getsockname()) for sock in server.sockets)
    print(f"Serving on {addrs}")
    async with server:
        await server.serve_forever()

if __name__ == "__main__":
    try:
        import uvloop
        asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
    except Exception:
        pass
    asyncio.run(main())
