"""Stable public business errors; internal reasons belong only in audit events."""


class BusinessError(ValueError):
    def __init__(self, code: str, message: str, *, internal_reason: str | None = None):
        super().__init__(message)
        self.code = code
        self.public_message = message
        self.internal_reason = internal_reason or code


class AuthorizationError(BusinessError):
    pass
