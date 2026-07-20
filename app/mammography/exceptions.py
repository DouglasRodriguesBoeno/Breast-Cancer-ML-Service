from __future__ import annotations


class MammographyError(Exception):
    def __init__(self, message: str, status_code: int = 400) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code


class MammographyValidationError(MammographyError):
    pass


class MammographyReadError(MammographyError):
    pass


class MammographyModelNotAvailableError(MammographyError):
    def __init__(self) -> None:
        super().__init__("Modelo de mamografia ainda nao disponivel.", status_code=503)

