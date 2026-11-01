"""OpenAI client factory and Georgian error messages. The key never leaves the server."""
from __future__ import annotations

import re

import openai

from .config import Settings

_KEY_RE = re.compile(r"(sk-[A-Za-z0-9_\-]{4})[A-Za-z0-9_\-*]+")


class MissingKeyError(Exception):
    pass


_WHERE_KEY = (
    "ადგილობრივად: ფაილში .env (პროექტის საქაღალდეში) ჩაწერეთ ხაზი OPENAI_API_KEY=თქვენი_გასაღები; "
    "ჰოსტინგზე: სერვისის Environment პარამეტრებში დაამატეთ ცვლადი OPENAI_API_KEY."
)
MISSING_KEY_MESSAGE = (
    "OpenAI API გასაღები არ არის მითითებული, ამიტომ AI პასუხი არ შეიქმნა. " + _WHERE_KEY +
    " ქვემოთ ნაჩვენებია მხოლოდ საკვანძო სიტყვებით ნაპოვნი წყაროები."
)


def get_client(settings: Settings) -> openai.OpenAI:
    if not settings.has_api_key:
        raise MissingKeyError(MISSING_KEY_MESSAGE)
    return openai.OpenAI(
        api_key=settings.openai_api_key,
        max_retries=settings.max_retries,
        timeout=settings.timeout_seconds,
    )


def sanitize(text: str) -> str:
    """Masks anything that looks like an API key before it is logged or shown."""
    return _KEY_RE.sub(r"\1…", text or "")


def describe_error(exc: Exception, settings: Settings | None = None, model: str | None = None) -> tuple[str, str]:
    """Returns (code, Georgian message) for an exception raised while calling OpenAI."""
    model = model or (settings.chat_model if settings else "")
    if isinstance(exc, MissingKeyError):
        return "missing_key", MISSING_KEY_MESSAGE
    if isinstance(exc, openai.AuthenticationError):
        return "auth", ("OpenAI-მ API გასაღები არ მიიღო (არასწორი, გაუქმებული ან არასრულად ჩაწერილი გასაღები). "
                        "შეამოწმეთ OPENAI_API_KEY. " + _WHERE_KEY)
    if isinstance(exc, openai.PermissionDeniedError):
        return "permission", (f"ანგარიშს ან პროექტს არ აქვს წვდომა მოდელზე „{model}“ ან ამ მოქმედებაზე. "
                              "შეამოწმეთ პროექტის უფლებები OpenAI-ის პანელში ან შეცვალეთ მოდელი .env-ში.")
    if isinstance(exc, openai.NotFoundError):
        return "model_not_found", (f"მოდელი „{model}“ ვერ მოიძებნა ან ამ ანგარიშისთვის მიუწვდომელია. "
                                   "შეცვალეთ OPENAI_CHAT_MODEL ან OPENAI_EMBEDDING_MODEL .env ფაილში.")
    if isinstance(exc, openai.RateLimitError):
        body = getattr(exc, "body", None) or {}
        code = (body.get("code") if isinstance(body, dict) else None) or getattr(exc, "code", None)
        if code == "insufficient_quota":
            return "quota", ("OpenAI ანგარიშზე კვოტა ან თანხა ამოიწურა. შეავსეთ ბალანსი ან შეამოწმეთ ლიმიტები "
                             "OpenAI-ის Billing გვერდზე.")
        return "rate_limit", ("OpenAI-ის მოთხოვნების ლიმიტი დროებით ამოიწურა (განმეორებითი მცდელობების შემდეგაც). "
                              "მოიცადეთ რამდენიმე წამი და სცადეთ თავიდან.")
    if isinstance(exc, openai.APITimeoutError):
        return "timeout", "OpenAI-მ დროულად ვერ უპასუხა. სცადეთ თავიდან ცოტა ხანში."
    if isinstance(exc, openai.APIConnectionError):
        return "connection", "OpenAI-სთან კავშირი ვერ დამყარდა. შეამოწმეთ ინტერნეტ-კავშირი."
    if isinstance(exc, openai.InternalServerError):
        return "server", "OpenAI-ის სერვერზე დროებითი შეცდომაა. სცადეთ თავიდან რამდენიმე წუთში."
    if isinstance(exc, openai.BadRequestError):
        msg = sanitize(getattr(exc, "message", str(exc)))
        return "bad_request", f"OpenAI-მ მოთხოვნა უარყო: {msg}"
    if isinstance(exc, openai.APIStatusError):
        return "api_error", f"OpenAI-ის შეცდომა (HTTP {exc.status_code}): {sanitize(str(exc))}"
    return "unexpected", f"მოულოდნელი შეცდომა: {sanitize(str(exc))}"
