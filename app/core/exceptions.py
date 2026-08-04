import traceback

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from app.utils.logger import logger


class AppException(Exception):
    """自定义业务异常"""

    def __init__(self, code: int, message: str, status_code: int = 400):
        self.code = code
        self.message = message
        self.status_code = status_code
        super().__init__(message)


def register_exception_handlers(app: FastAPI):
    """注册全局异常处理器"""

    @app.exception_handler(AppException)
    async def app_exception_handler(request: Request, exc: AppException):
        logger.log(
            "error",
            {
                "operation": "app_exception",
                "code": exc.code,
                "error": exc.message,
                "desc": f"业务异常: {exc.message}",
            },
        )
        return JSONResponse(
            status_code=exc.status_code,
            content={"code": exc.code, "message": exc.message, "data": None},
        )

    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException):
        logger.log(
            "error",
            {
                "operation": "http_exception",
                "status_code": exc.status_code,
                "error": exc.detail,
                "desc": f"HTTP 异常: {exc.detail}",
            },
        )
        return JSONResponse(
            status_code=exc.status_code,
            content={"code": exc.status_code, "message": exc.detail, "data": None},
        )

    @app.exception_handler(ValueError)
    async def value_error_handler(request: Request, exc: ValueError):
        logger.log(
            "error",
            {
                "operation": "value_error",
                "error": str(exc),
                "desc": f"参数错误: {str(exc)}",
            },
        )
        return JSONResponse(
            status_code=400, content={"code": 400, "message": str(exc), "data": None}
        )

    @app.exception_handler(RuntimeError)
    async def runtime_error_handler(request: Request, exc: RuntimeError):
        logger.log(
            "error",
            {
                "operation": "runtime_error",
                "error": str(exc),
                "traceback": traceback.format_exc(),
                "desc": f"运行时错误: {str(exc)}",
            },
        )
        return JSONResponse(
            status_code=500,
            content={"code": 500, "message": "服务内部错误，请稍后重试", "data": None},
        )

    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        logger.log(
            "error",
            {
                "operation": "global_exception",
                "error": str(exc),
                "traceback": traceback.format_exc(),
                "desc": f"未捕获异常: {str(exc)}",
            },
        )
        return JSONResponse(
            status_code=500,
            content={"code": 500, "message": "服务内部错误，请稍后重试", "data": None},
        )
