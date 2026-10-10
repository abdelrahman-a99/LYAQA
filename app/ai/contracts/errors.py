"""Processing failures distinct from valid conversational outcomes."""


class ProcessingError(Exception):
    def __init__(
        self,
        message: str,
        *,
        code: str,
        status_code: int,
        retryable: bool = False,
    ):
        super().__init__(message)
        self.code = code
        self.status_code = status_code
        self.retryable = retryable
