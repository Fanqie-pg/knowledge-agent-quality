"""为模型调用提供可测试的有限重试能力。"""

import logging
import time
from collections.abc import Callable


logger = logging.getLogger(__name__)


class ModelUnavailableError(RuntimeError):
    """表示模型服务经过有限重试后仍不可用。"""


class RetryingChatModel:
    """包装聊天模型，并在瞬时异常时执行有限重试。"""

    def __init__(
        self,
        model,
        max_attempts: int = 2,
        retry_delay_seconds: float = 0,
        sleep_fn: Callable[[float], None] = time.sleep,
        retryable_exceptions: tuple[type[Exception], ...] = (
            TimeoutError,
            ConnectionError,
        ),
    ):
        if max_attempts <= 0:
            raise ValueError("max_attempts必须大于0")
        if retry_delay_seconds < 0:
            raise ValueError("retry_delay_seconds不能小于0")
        if not retryable_exceptions:
            raise ValueError("retryable_exceptions不能为空")

        self.model = model
        self.max_attempts = max_attempts
        self.retry_delay_seconds = retry_delay_seconds
        self.sleep_fn = sleep_fn
        self.retryable_exceptions = retryable_exceptions

    def invoke(self, model_input, *args, **kwargs):
        for attempt in range(1, self.max_attempts + 1):
            try:
                return self.model.invoke(model_input, *args, **kwargs)
            except self.retryable_exceptions as exc:
                if attempt == self.max_attempts:
                    logger.error(
                        "模型调用重试耗尽："
                        "attempt=%s/%s, error_type=%s",
                        attempt,
                        self.max_attempts,
                        type(exc).__name__,
                    )
                    raise ModelUnavailableError("模型服务暂时不可用") from exc

                logger.warning(
                    "模型调用失败，准备重试："
                    "attempt=%s/%s, error_type=%s",
                    attempt,
                    self.max_attempts,
                    type(exc).__name__,
                )

                delay = (
                    self.retry_delay_seconds
                    * (2 ** (attempt - 1))
                )
                if delay > 0:
                    self.sleep_fn(delay)

    def bind_tools(self, tools, **kwargs):
        bound_model = self.model.bind_tools(tools, **kwargs)

        return RetryingChatModel(
            model=bound_model,
            max_attempts=self.max_attempts,
            retry_delay_seconds=self.retry_delay_seconds,
            sleep_fn=self.sleep_fn,
            retryable_exceptions=self.retryable_exceptions,
        )