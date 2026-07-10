import logging

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.config import allowed_cors_origins, settings

logger = logging.getLogger("visioninspect.errors")


def error_response(
    *,
    request: Request,
    status_code: int,
    code: str,
    message: str,
    details: object | None = None,
) -> JSONResponse:
    request_id = getattr(request.state, "request_id", None)
    payload = {
        "error": {
            "code": code,
            "message": message,
            "request_id": request_id,
        }
    }
    if details is not None:
        payload["error"]["details"] = details
    response = JSONResponse(status_code=status_code, content=payload)
    origin = request.headers.get("origin")
    if origin and origin in allowed_cors_origins():
        response.headers["Access-Control-Allow-Origin"] = origin
        response.headers["Access-Control-Allow-Credentials"] = "true"
        response.headers["Vary"] = "Origin"
    return response


def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
        detail = exc.detail
        message = detail if isinstance(detail, str) else "Request could not be completed"
        return error_response(
            request=request,
            status_code=exc.status_code,
            code="http_error",
            message=message,
            details=None if isinstance(detail, str) else detail,
        )

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
        details = [
            {
                "field": ".".join(str(part) for part in error.get("loc", [])),
                "message": error.get("msg", "Invalid value"),
            }
            for error in exc.errors()
        ]
        return error_response(
            request=request,
            status_code=422,
            code="validation_error",
            message="Please check the submitted data and try again",
            details=details,
        )

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        request_id = getattr(request.state, "request_id", None)
        logger.exception("Unhandled backend error", extra={"request_id": request_id})
        details = str(exc) if settings.environment.lower() != "production" else None
        return error_response(
            request=request,
            status_code=500,
            code="internal_server_error",
            message="Something went wrong while processing the request",
            details=details,
        )
