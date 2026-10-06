"""Bound request bodies before JSON/multipart parsers allocate larger payloads."""

from starlette.responses import JSONResponse


class RequestSizeLimit:
    def __init__(self, app, limit=2 * 1024 * 1024 + 16384):
        self.app, self.limit = app, limit

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope["method"] not in ("POST", "PUT", "PATCH"):
            return await self.app(scope, receive, send)
        messages, size = [], 0
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                return
            size += len(message.get("body", b""))
            if size > self.limit:
                return await JSONResponse(
                    {"detail": "Request exceeds the 2 MiB upload limit"},
                    status_code=413,
                )(scope, receive, send)
            messages.append(message)
            if not message.get("more_body", False):
                break
        index = 0

        async def replay():
            nonlocal index
            if index < len(messages):
                result = messages[index]
                index += 1
                return result
            return await receive()

        await self.app(scope, replay, send)
