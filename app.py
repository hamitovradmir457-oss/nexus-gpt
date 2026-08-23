r"""
🤖 Telegram AI Bot
==================
- Чат-модели через OpenRouter, объединённые в 4 КАТЕГОРИИ (Кодинг, Универсальные,
  Быстрые, Творчество). Пользователь выбирает категорию, а не конкретную
  модель — бот сам перебирает модели ВНУТРИ выбранной категории при перегрузке
  (HTTP 429), без выхода за её пределы. Ответ стримится в реальном времени.
- STT через Mistral Voxtral (голос / аудио / видео / видеокружки)
- Обязательная подписка через BotoHub
- Баланс токенов: списание за объём ответа, пополнение за Stars / ₽
- Полная поддержка Telegram Rich Messages (Bot API 10.1): заголовки, списки,
  таблицы, цитаты, спойлеры, формулы (LaTeX), code-блоки, ссылки,
  зачёркнутый/подчёркнутый текст, вложенные списки, блочные спойлеры, с авто-
  разбивкой на несколько сообщений при превышении лимитов (блоков И байт) и
  fallback на HTML (<b>, <i>, <s>, <u>, <tg-spoiler>, <a>) / plain text.
- LaTeX-команды: \cdot, \bullet -> •; полный набор греческих букв, стрелок,
  операторов и символов множеств -> юникод (применяется везде: внутри формул
  $...$/$$...$$ и в обычном тексте, независимо от движка рендеринга).
- Фото / изображения: в любой категории автоматически уходят на vision-модели
  (Gemma 4 31B, Nemotron VL / Omni) — отдельную категорию выбирать не нужно.

pip install aiogram>=3.29.0 httpx "telegramify-markdown>=1.0"
export BOT_TOKEN="..."
python bot.py
"""

from __future__ import annotations

import asyncio
import base64
import glob
import io
import json
import logging
import os
import re
import sqlite3
import sys
import time
from collections import Counter
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from os import getenv
from typing import Optional

import httpx

from aiogram import Bot, Dispatcher, F, Router, html
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode
from aiogram.exceptions import TelegramBadRequest, TelegramConflictError
from aiogram.filters import Command, CommandStart
from aiogram.types import (
    BufferedInputFile,
    ErrorEvent,
    FSInputFile,
    CallbackQuery,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    InputMediaPhoto,
    LabeledPrice,
    Message,
    PreCheckoutQuery,
    ReplyKeyboardRemove,
)
from aiogram.utils.keyboard import InlineKeyboardBuilder

try:
    import telegramify_markdown
    TELEGRAMIFY_AVAILABLE = hasattr(telegramify_markdown, "telegramify_rich")
    if not TELEGRAMIFY_AVAILABLE:
        logging.warning(
            "telegramify_markdown установлен, но без telegramify_rich() — "
            "нужна версия >=1.0 (pip install -U telegramify-markdown). "
            "Формулы будут рендериться через собственный конвертер."
        )
except ImportError:
    telegramify_markdown = None
    TELEGRAMIFY_AVAILABLE = False


# ══════════════════════════════════════════════════════════════
# КОНФИГУРАЦИЯ
# ══════════════════════════════════════════════════════════════

# Все секреты вынесены в файл .env рядом со скриптом (НЕ коммитьте его в git!).
# Нужен python-dotenv: pip install python-dotenv. Без него читаются только
# обычные переменные окружения (export BOT_TOKEN=... и т.д.).
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    logging.warning(
        "python-dotenv не установлен — файл .env не читается. "
        "Установите: pip install python-dotenv"
    )

BOT_TOKEN: str = getenv("BOT_TOKEN", "")
if not BOT_TOKEN:
    sys.exit("⛔ BOT_TOKEN не задан. Заполните файл .env рядом со скриптом или переменные окружения.")

_raw_admins = getenv("ADMIN_IDS", "")
ADMIN_IDS: set[int] = {int(x) for x in _raw_admins.split(",") if x.strip().isdigit()}

# ── Anthropic (само-редактирование кода бота через Claude) ──────
# Бот умеет переписывать собственный исходник по команде админа /selfedit
# <что изменить>: читает свой файл, отправляет его Claude через Anthropic API,
# получает полный обновлённый файл, проверяет синтаксис, делает бэкап,
# перезаписывает себя и перезапускается (панель play2go поднимет заново).
# Запросы идут через EchoGate (ECHOGATE_KEY ниже) — отдельный Anthropic-ключ
# и пакет anthropic НЕ нужны. Модель по умолчанию — claude-fable-5 (сильнейшая
# для кода); переопределить можно переменной SELFEDIT_MODEL.
SELFEDIT_MODEL: str = getenv("SELFEDIT_MODEL", "claude-fable-5")

# ── play2go / Pterodactyl (перезапуск сервера как кнопкой в консоли) ──
# Если задан ключ клиентского API панели — рестарт идёт через её API
# (надёжно, как кнопка в консоли). Если НЕ задан — бот просто выходит, и
# панель поднимает процесс сама (авто-рестарт при остановке). Ключ (ptlc_...)
# создаётся в панели: Account → API Credentials.
PANEL_URL: str = getenv("PANEL_URL", "https://control.play2go.cloud").rstrip("/")
PANEL_API_KEY: str = getenv("PANEL_API_KEY", "")
SERVER_ID: str = getenv("SERVER_ID", "51866ae7")

OPENROUTER_KEY: str = getenv("OPENROUTER_KEY", "")
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

# ── Featherless (второй провайдер моделей, OpenAI-совместимый API) ──
# Отдельный сервис со СВОИМ ключом (НЕ ключ OpenRouter). Взять на
# https://featherless.ai/account/api-keys. Модели с provider="featherless"
# (см. ModelInfo) уходят сюда. Ключ впишите вторым аргументом getenv:
FEATHERLESS_KEY: str = getenv("FEATHERLESS_KEY", "")
FEATHERLESS_URL = "https://api.featherless.ai/v1/chat/completions"

# ── FreeTheAi (третий провайдер, OpenAI-совместимый API) ───────────
# Свой ключ (НЕ OpenRouter/Featherless). Выдаётся в Discord-сервере
# FreeTheAi командой /signup, и ОБЯЗАТЕЛЬНО раз в сутки (по UTC) нужно
# сделать /checkin (ввести ключ + решить капчу), иначе запросы падают
# с 403 daily_checkin_required. Модели с provider="freetheai" уходят сюда.
# Впишите ключ вторым аргументом getenv вместо "":
FREETHEAI_KEY: str = getenv("FREETHEAI_KEY", "")
FREETHEAI_URL = "https://api.freetheai.xyz/v1/chat/completions"

# ── EchoGate / VEXZY (четвёртый провайдер, OpenAI-совместимый API) ──
# Ключ формата forge-... выдаёт Telegram-бот @vexzy_bot (раздел «🔑 API»).
# Модели с provider="echogate" уходят сюда. Дневной лимит кредитов
# сбрасывается в 00:00 МСК (при исчерпании API отвечает 402).
# Впишите ключ в .env: ECHOGATE_KEY=forge-...
ECHOGATE_KEY: str = getenv("ECHOGATE_KEY", "")
ECHOGATE_URL = "https://api.echogate.one/v1/chat/completions"

# Публичный сайт и серверный статус моделей. /api/model-health проксируется
# через Vercel, поэтому бот и сайт получают одинаковые данные FreeTheAI.
SITE_URL = "https://www.gptly.top/"
MODEL_HEALTH_PAGE_URL = "https://www.gptly.top/#health"
MODEL_HEALTH_API_URL = "https://www.gptly.top/api/model-health"
MODEL_HEALTH_CACHE_TTL = 30.0

MISTRAL_KEY: str = getenv("MISTRAL_KEY", "")
MISTRAL_STT_URL = "https://api.mistral.ai/v1/audio/transcriptions"
MISTRAL_STT_MODEL = "voxtral-mini-latest"

# ── Platega (приём оплаты в рублях) ────────────────────────────
# Платёжный провайдер: оплата пакетов/Premium за рубли (СБП, карта, USDT) в
# дополнение к Telegram Stars. Доступы (MerchantId + Secret) приходят от
# менеджера на почту и есть в ЛК → Настройки. Впишите их вторыми аргументами
# getenv (или задайте через переменные окружения на хостинге).
#
# Архитектура без вебхука: бот создаёт транзакцию (POST /transaction/process),
# даёт пользователю ссылку на оплату, а по кнопке «Я оплатил» сам проверяет
# статус (GET /transaction/{id}). Начисление идемпотентное — см. platega_tx.
PLATEGA_MERCHANT_ID: str = getenv("PLATEGA_MERCHANT_ID", "")
PLATEGA_SECRET: str = getenv("PLATEGA_SECRET", "")
PLATEGA_BASE_URL = "https://app.platega.io"
# Ссылки редиректа после оплаты (не критичны — оплата подтверждается кнопкой).
# По умолчанию ведут в Telegram; при желании укажите ссылку на своего бота.
PLATEGA_RETURN_URL: str = getenv("PLATEGA_RETURN_URL", "https://t.me")
PLATEGA_FAILED_URL: str = getenv("PLATEGA_FAILED_URL", "https://t.me")

# Коды методов оплаты Platega (поле paymentMethod). Из PaymentMethodInt:
#   2 — СБП (QR) + SberPay | 3 — ЕРИП | 11 — карта | 12 — межд. | 13 — крипта
PLATEGA_METHODS: dict[str, dict] = {
    "sbp":    {"code": 2,  "emoji": "🟢", "name": "СБП / QR"},
    "card":   {"code": 11, "emoji": "💳", "name": "Банковская карта"},
    "crypto": {"code": 13, "emoji": "🪙", "name": "Криптовалюта (USDT)"},
}


def platega_enabled() -> bool:
    """True, если Platega сконфигурирована (заданы MerchantId и Secret)."""
    return bool(PLATEGA_MERCHANT_ID and PLATEGA_SECRET)

# ── BotoHub Views (реклама) ────────────────────────────────────
BOTOHUB_TOKEN: str = getenv("BOTOHUB_TOKEN", "")
BOTOHUB_URL = "https://views.botohub.me/ad/SendPost"
BOTOHUB_AD_EVERY: int = max(1, int(getenv("BOTOHUB_AD_EVERY", "5")))  # 0 = деление на ноль


# ── BotoHub Биржа ОП / Продвинутая интеграция (обязательная подписка) ──
# ВНИМАНИЕ: это ОТДЕЛЬНЫЙ сервис от Views выше. Эндпоинт botohub.me/get-tasks
# возвращает {"tasks": [ссылки], "completed": bool, "skip": bool} и требует
# СВОЙ токен биржи ОП (НЕ токен Views 95abd7b8-...!). Токен передаётся в
# заголовке Auth. Взять токен у @botohubbot в личном кабинете.
#
# В отличие от старого botohub.me/send (BotoHub сам слал пост), здесь мы
# ПОЛУЧАЕМ список ссылок на спонсоров и САМИ формируем и отправляем сообщение
# с кнопками-подписками и кнопкой «✅ Проверить». Спонсоры закрепляются за
# пользователем на 3 мин; при повторном запросе возвращаются только
# невыполненные.
#
# КУДА ВСТАВИТЬ ТОКЕН: в файл .env рядом со скриптом (BOTOHUB_OP_TOKEN=...).
# Пока пусто — гейт fail-open: пропускает всех, бот работает как раньше.
BOTOHUB_OP_TOKEN: str = getenv("BOTOHUB_OP_TOKEN", "")
BOTOHUB_OP_URL: str = getenv("BOTOHUB_OP_URL", "https://botohub.me/get-tasks")
# Сколько ответов новый пользователь получает бесплатно (без ОП), чтобы зацепить.
# После них ОП требуется раз в день (у BotoHub каждый день новые спонсоры).
OP_FREE_ANSWERS: int = int(getenv("OP_FREE_ANSWERS", "1"))


# ── ТОКЕН-ЭКОНОМИКА ────────────────────────────────────────────
# Баланс токенов вместо дневных лимитов: пользователь пополняет баланс
# и платит за фактический объём (вопрос + контекст + ответ).
TOKENS_PER_RUB = 600          # курс: 1 ₽ = 600 токенов
TOKENS_PER_STAR = 720         # 1 ⭐ ≈ 1.2 ₽ → 720 токенов
MIN_TOKENS_SPEND = 400        # минимальное списание за один ответ
CHARS_PER_TOKEN_EST = 3       # оценка: 1 токен ≈ 3 символа текста
WELCOME_TOKENS = 10_000       # разовый стартовый баланс новичку
PREMIUM_SPEND_DISCOUNT = 0.8  # Premium: списание на 20% дешевле
TOPUP_MIN_STARS, TOPUP_MAX_STARS = 10, 100_000   # своя сумма, ⭐
TOPUP_MIN_RUB, TOPUP_MAX_RUB = 50, 100_000       # своя сумма, ₽

FREE_LIMIT = 10   # бесплатных запросов в день (учтён ежедневный бонус и рефералка)

MAX_HISTORY = 20
MAX_TOKENS = 4096
MAX_TG = 4096
# Сколько последних покупок держим в журнале. Таблица перезаписывается целиком
# при каждом сохранении, поэтому неограниченный рост = растущая пауза для всех.
MAX_PURCHASES = int(getenv("MAX_PURCHASES", "5000"))

# ── ДОКУМЕНТЫ И ДЛИННЫЕ ОТВЕТЫ ─────────────────────────────────
# Если готовый ответ длиннее порога — отдаём его .md-файлом, а не десятком
# сообщений (так удобнее читать/сохранять большие тексты и код).
SEND_AS_FILE_THRESHOLD = 9000
# Сколько символов текста из присланного документа отправляем модели
# (защита от переполнения контекста и лишних токенов).
MAX_DOC_CHARS = 30000
# Расширения, которые читаем как обычный текст (код, разметка, данные).
TEXT_DOC_EXTS = {
    ".txt", ".md", ".markdown", ".csv", ".tsv", ".json", ".xml", ".yaml", ".yml",
    ".ini", ".cfg", ".log", ".html", ".htm", ".rtf", ".srt", ".vtt",
    ".py", ".js", ".ts", ".jsx", ".tsx", ".java", ".c", ".h", ".cpp", ".cc",
    ".cs", ".go", ".rs", ".rb", ".php", ".swift", ".kt", ".sql", ".sh", ".bat",
    ".css", ".scss", ".r", ".m", ".lua", ".pl", ".toml", ".env",
}

# ── НАГРАДЫ (реферальная система и ежедневный бонус) ───────────
# Начисляются в usage["bought"] — тот же механизм, что и купленные пакеты,
# поэтому расходуются вместе с дневным лимитом. Меняются одной строкой.
REF_REWARD = 20_000    # токенов пригласившему за каждого друга
REF_WELCOME = 10_000   # токенов новичку, пришедшему по ссылке
DAILY_BONUS = 10_000   # токенов за ежедневный бонус (1-й день стрика)
DAILY_BONUS_STREAK = (DAILY_BONUS, 12_000, 15_000, 18_000, 20_000)  # бонус по дням стрика; дальше — потолок
REF_ACTIVATION_SPEND = 5_000  # сколько токенов должен потратить приглашённый, чтобы реферер получил награду

# ── Реферальные уровни: всего приглашённых друзей -> разовый бонус токенов ──
REF_MILESTONES: dict[int, int] = {5: 50_000, 10: 150_000, 25: 500_000}

# ── Пробный Premium: разовый бесплатный период (0 = отключить триал) ──
TRIAL_DAYS = 3

# ── Свой системный промпт (Premium): максимальная длина ────────
CUSTOM_PROMPT_MAX_LEN = 500

# ── ДЛИНА ОТВЕТА (verbosity) ───────────────────────────────────
# Пользователь выбирает, насколько развёрнуто отвечает бот. Управляет и
# потолком токенов, и подсказкой в системном промпте.
VERBOSITY_TOKENS = {"short": 800, "medium": 2048, "long": 4096}
VERBOSITY_LABELS = {"short": "✂️ Кратко", "medium": "📄 Средне", "long": "📚 Подробно"}
VERBOSITY_HINT = {
    "short": "Отвечай максимально кратко — 1–3 предложения, только суть.",
    "medium": "",  # обычное поведение, без доп-инструкций
    "long": "Отвечай развёрнуто и подробно, с примерами и пояснениями.",
}
DEFAULT_VERBOSITY = "medium"

# ── БАННЕРЫ МЕНЮ ───────────────────────────────────────────────
# Папка с картинками экранов (img/welcome.png, img/cat_menu.png,
# img/cat_<категория>.png). Если файла нет — тихий fallback на обычный
# текст, ничего не ломается. Генератор: make_banners.py.
MENU_IMAGE_DIR = getenv("MENU_IMAGE_DIR", "img")
# Картинка отправляется в _send_welcome. Если файла нет — тихий fallback
# на обычный текст. Путь переопределяется через переменную окружения.
WELCOME_IMAGE = getenv("WELCOME_IMAGE", os.path.join(MENU_IMAGE_DIR, "welcome.png"))

# Таймаут одного запроса к модели. Раньше был 120с: при переборе нескольких
# «лежащих» моделей пользователь ждал минутами. Затем 45с — но саппорт FreeTheAi
# советует таймаут повыше: reasoning-модели (GLM/DeepSeek) долго «думают» и на
# 45с ответ обрезался по таймауту. Компромисс — 60с: даём подумать, но не ждём
# минутами. Ошибки провайдера (503/502) прилетают мгновенно и таймаут не тратят.
REQUEST_TIMEOUT = 60

# Общий дедлайн на ВЕСЬ ход (перебор всех моделей категории), а не на один
# запрос. Без него 8 моделей × REQUEST_TIMEOUT + ретраи шлюза давали до ~24 мин
# ожидания, причём статус «Думаю...» в ветке таймаута не обновлялся: человек
# молча смотрел на индикатор и уходил. 90с = хватает reasoning-моделям на
# 1-2 попытки, но не превращается в бесконечность.
AI_TURN_TIMEOUT = int(getenv("AI_TURN_TIMEOUT", "90"))

# Защита общих ключей провайдеров. Ключи бесплатных тарифов лимитированы по
# частоте и делятся между ВСЕМИ пользователями бота: один активный человек,
# отправляя вопросы подряд (каждый — перебор до 8 моделей с ретраями), способен
# выжечь квоту, и тогда бот перестаёт отвечать всем сразу. Симптом выглядит как
# «провайдеры упали», а не как «нам нужен троттлинг».
#
# AI_MAX_CONCURRENT — сколько запросов к моделям может идти одновременно по всему
# боту. USER_COOLDOWN_SEC — минимальный интервал между вопросами одного человека.
AI_MAX_CONCURRENT = max(1, int(getenv("AI_MAX_CONCURRENT", "4")))  # 0 = мёртвый семафор
USER_COOLDOWN_SEC = float(getenv("USER_COOLDOWN_SEC", "3"))

RICH_MAX_BYTES = 32768
RICH_MAX_BLOCKS = 500


# ── ПАКЕТЫ ТОКЕНОВ (магазин) ─────────────────────────────────
# Несколько пакетов с прогрессивной скидкой за токен. Оплата идёт
# через УЖЕ существующий механизм Telegram Stars (cmd_buy → invoice →
# pre_checkout → on_payment). Меняется только количество начисляемых
# токенов и цена пакета — сам платёжный поток не трогаем.

@dataclass(frozen=True)
class RequestPack:
    key: str          # внутренний ключ (уходит в payload инвойса)
    tokens: int   # сколько токенов начисляется
    stars: int        # цена в Telegram Stars (XTR)
    emoji: str        # эмодзи пакета
    note: str = ""    # короткая подпись-выгода
    rub: int = 0      # цена в рублях (для оплаты через Platega)


REQUEST_PACKS: dict[str, RequestPack] = {
    "p50":   RequestPack("p50",    36_000,   50, "⭐", note="",                    rub=60),
    "p200":  RequestPack("p200",  135_000,  180, "🔥", note="выгодно",             rub=219),
    "p500":  RequestPack("p500",  300_000,  400, "💎", note="лучшая цена",         rub=489),
    "p1500": RequestPack("p1500", 800_000, 1050, "👑", note="максимальная выгода", rub=1290),
}
DEFAULT_PACK = "p50"   # совместимость со старым payload "ai_pack"


def _pack_from_payload(payload: str) -> Optional[RequestPack]:
    """Определяет пакет по payload инвойса. None — пакет НЕ найден.

    Новый формат: "ai_pack:<key>". Старый формат "ai_pack" (одиночная
    покупка до появления магазина) трактуется как пакет по умолчанию —
    так ранее выставленные счета продолжат корректно зачисляться.
    Раньше ЛЮБОЙ незнакомый ключ молча заменялся дефолтным пакетом:
    заплативший за p1500 получал токены p50.
    """
    if payload and ":" in payload:
        return REQUEST_PACKS.get(payload.split(":", 1)[1])
    return REQUEST_PACKS[DEFAULT_PACK]


# ── СКИДКИ / АКЦИИ (глобальная распродажа, включается из админки) ──

def sale_percent() -> int:
    """Активный процент скидки (0 — скидки нет или она истекла)."""
    if not sale_info:
        return 0
    until = sale_info.get("until")
    if until:
        try:
            if date.today() > date.fromisoformat(until):
                return 0
        except Exception:
            return 0
    p = int(sale_info.get("percent", 0))
    return p if 0 < p < 100 else 0


def _disc(amount: int) -> int:
    """Цена с учётом активной скидки (не меньше 1)."""
    p = sale_percent()
    if not p:
        return amount
    return max(1, round(amount * (100 - p) / 100))


def _sale_banner() -> str:
    """Строка-баннер акции для магазина ('' — если скидки нет)."""
    p = sale_percent()
    if not p:
        return ""
    till = ""
    until = sale_info.get("until")
    if until:
        try:
            till = f" до {date.fromisoformat(until).strftime('%d.%m')}"
        except Exception:
            pass
    return f"🏷 <b>АКЦИЯ −{p}%</b> на все покупки{till}!\n"


# ── ТАРИФЫ (расширяемая заготовка под будущий Premium) ─────────
# Premium ПОКА ЕЩЁ НЕ реализован и нигде не включается. Структура заведена
# заранее (требование «подготовить архитектуру»), чтобы позже можно
# было добавить подписку, отключение рекламы и расширенные лимиты, НЕ
# трогая бизнес-логику хендлеров: достаточно добавить план в PLANS и
# проставлять user_plans[uid]. Сейчас у всех — план "free".

@dataclass(frozen=True)
class Plan:
    key: str
    name: str
    daily_limit: int   # бесплатных запросов в день
    show_ads: bool      # показывать ли рекламу


UNLIMITED = float("inf")    # «бесконечный» дневной лимит (для админов)
PREMIUM_LIMIT = 1000         # дневной лимит на тарифе Premium
PREMIUM_DAYS = 30            # длительность Premium при покупке
PREMIUM_PRICE = 300          # цена Premium в Telegram Stars (XTR)
PREMIUM_PRICE_RUB = 590      # цена Premium в рублях (оплата через Platega)

PLANS: dict[str, Plan] = {
    "free":    Plan("free",    "Бесплатный", FREE_LIMIT,    True),
    "premium": Plan("premium", "Premium ⭐", PREMIUM_LIMIT, False),
    "admin":   Plan("admin",   "Админ ∞",    UNLIMITED,     False),
}
DEFAULT_PLAN = "free"
user_plans: dict[int, str] = {}              # uid -> ключ тарифа (ручной override)
premium_until: dict[int, datetime] = {}      # uid -> когда истекает Premium (UTC)

# Категории, доступные только Premium-пользователям (и админам).
PREMIUM_CATEGORIES: set[str] = {"fast"}

# Категории «в разработке»: показываются в меню, но НЕ открываются.
# При нажатии пользователь видит уведомление, что раздел ещё не готов.
WIP_CATEGORIES: set[str] = set()
WIP_NOTICE = "🛠 Эта категория ещё в разработке и пока недоступна."


# ── ПЕРСОНЫ (характер/роль ассистента) ─────────────────────────
# Пользователь выбирает «характер» бота. Технически это просто разный
# system_extra, который дописывается к системному промпту в ask_ai —
# поверх добавки категории. "default" = без изменений (обычный ассистент).
@dataclass(frozen=True)
class Persona:
    key: str
    name: str
    emoji: str
    system_extra: str


PERSONAS: dict[str, Persona] = {
    "default": Persona("default", "Обычный", "🤖", ""),
    "expert": Persona(
        "expert", "Эксперт", "🎓",
        "Ты — строгий эксперт. Отвечай точно, структурировано и по делу, "
        "опирайся на факты, избегай лишней воды и оговорок.",
    ),
    "friendly": Persona(
        "friendly", "Дружелюбный", "😊",
        "Ты — тёплый и дружелюбный собеседник. Поддерживай, объясняй просто "
        "и доброжелательно, уместно используй эмодзи.",
    ),
    "humor": Persona(
        "humor", "С юмором", "😏",
        "Ты остроумный собеседник. Отвечай по существу, но с лёгким уместным "
        "юмором и живыми формулировками. Не переусердствуй с шутками.",
    ),
    "tech": Persona(
        "tech", "Технический", "🧑‍💻",
        "Ты — технический специалист. Отвечай сжато и предметно, используй "
        "точные термины, код давай в блоках с указанием языка.",
    ),
}
DEFAULT_PERSONA = "default"


def category_wip_for(uid: int, cat_key: str) -> bool:
    """True, если категория «в разработке» И пользователь не админ.

    Админам раздел доступен для тестирования; остальные видят его как WIP.
    """
    return cat_key in WIP_CATEGORIES and uid not in ADMIN_IDS


def premium_active(uid: int) -> bool:
    """True, если у пользователя есть действующая Premium-подписка."""
    until = premium_until.get(uid)
    return until is not None and until > datetime.now(timezone.utc)


def grant_premium(uid: int, days: int = PREMIUM_DAYS) -> datetime:
    """Выдаёт/продлевает Premium. Возвращает дату окончания (UTC)."""
    base = datetime.now(timezone.utc)
    current = premium_until.get(uid)
    if current and current > base:
        base = current   # продлеваем от текущей даты окончания, не теряя остаток
    premium_until[uid] = base + timedelta(days=days)
    return premium_until[uid]


def user_plan(uid: int) -> Plan:
    # Админы всегда на безлимитном тарифе с Premium-доступом.
    if uid in ADMIN_IDS:
        return PLANS["admin"]
    if premium_active(uid):
        return PLANS["premium"]
    return PLANS.get(user_plans.get(uid, DEFAULT_PLAN), PLANS[DEFAULT_PLAN])


# ── ПЕРСОНАЛИЗАЦИЯ (длина ответа / персона) ────────────────────

def user_verbosity_key(uid: int) -> str:
    """Ключ выбранной длины ответа (или дефолт)."""
    key = user_verbosity.get(uid, DEFAULT_VERBOSITY)
    return key if key in VERBOSITY_TOKENS else DEFAULT_VERBOSITY


def user_persona_obj(uid: int) -> Persona:
    """Выбранная персона пользователя (или дефолтная)."""
    return PERSONAS.get(user_persona.get(uid, DEFAULT_PERSONA), PERSONAS[DEFAULT_PERSONA])


# ── НАГРАДЫ (запросы начисляются в usage["bought"]) ────────────

def _grant_requests(uid: int, amount: int) -> None:
    """Начисляет пользователю дополнительные запросы (как купленный пакет)."""
    _get_usage(uid)["bought"] += amount


def _take_requests(uid: int, amount: int) -> None:
    """Списывает купленные/бонусные запросы (возвраты, админ-правки).

    Списываем не больше остатка: раньше «остаток» вливался в u["used"] и
    раздувал дневную активность, которой верил антиабуз ежедневного бонуса
    (_bonus_requires_activity). lifetime_spent здесь тоже не трогаем:
    это счётчик РЕАЛЬНО потраченного, возврат его не отменяет.
    """
    u = _get_usage(uid)
    u["bought"] = max(0, int(u["bought"]) - max(0, int(amount)))


def can_claim_daily_bonus(uid: int) -> bool:
    """True, если сегодня ежедневный бонус ещё не получали."""
    return daily_bonus_date.get(uid) != _today_iso()


def _bonus_requires_activity(uid: int) -> bool:
    """Антиабуз: требовать хотя бы один реальный запрос сегодня перед бонусом.

    Админы и те, у кого баланса не хватает даже на один запрос, освобождены
    (для них бонус — спасательный круг). Скрипту-ферме придётся жечь свои же
    токены на запросы, чтобы фармить токены — смысл теряется.
    """
    if uid in ADMIN_IDS:
        return False
    if remaining(uid) < MIN_TOKENS_SPEND:
        return False
    return int(_get_usage(uid)["used"]) <= 0


def _daily_bonus_amount(streak: int) -> int:
    """Размер бонуса для дня стрика (1-based) с потолком."""
    idx = max(0, min(int(streak) - 1, len(DAILY_BONUS_STREAK) - 1))
    return DAILY_BONUS_STREAK[idx]


def _valid_referrer(new_uid: int, ref_uid: int) -> bool:
    """Проверяет, что реферал легитимен: не сам себя, реферер существует как
    известный пользователь и новичок ещё никем не приглашён."""
    if ref_uid == new_uid:
        return False
    if new_uid in invited_by:          # уже засчитан ранее — повторно нельзя
        return False
    if ref_uid not in user_stats:      # приглашающий должен быть реальным юзером бота
        return False
    return True


async def _award_referral_if_any(bot: Bot, new_uid: int) -> None:
    """Регистрирует реферала после принятия соглашения новичком.

    Новичку сразу — REF_WELCOME. Награда пригласившему (REF_REWARD) отложена:
    начислится, когда новичок реально потратит REF_ACTIVATION_SPEND токенов
    (см. _maybe_activate_referral). Это обесценивает фермы пустых аккаунтов.
    Повторное срабатывание исключено отметкой invited_by[new_uid].
    """
    # get, не pop: если реферер невалиден (например, удалён уборкой за
    # неактивность), запись остаётся — вдруг он вернётся. Раньше pop стоял
    # до проверки и ссылка сгорала молча и навсегда.
    ref_uid = pending_referral.get(new_uid)
    if ref_uid is None or not _valid_referrer(new_uid, ref_uid):
        return
    pending_referral.pop(new_uid, None)

    invited_by[new_uid] = ref_uid
    ref_pending_award[new_uid] = ref_uid     # награда реферера ждёт активности новичка
    _grant_requests(new_uid, REF_WELCOME)    # приветственный бонус новичку — сразу
    save_state()

    # Уведомляем пригласившего (best-effort — он мог заблокировать бота).
    try:
        await bot.send_message(
            ref_uid,
            f"🎉 <b>По вашей ссылке пришёл новый пользователь!</b>\n\n"
            f"➕ Награда <b>{fmt_tokens(REF_REWARD)}</b> токенов придёт, как только он "
            f"освоится и потратит первые <b>{fmt_tokens(REF_ACTIVATION_SPEND)}</b> токенов.",
        )
    except Exception:
        pass


async def _maybe_activate_referral(bot: Bot, uid: int) -> None:
    """Начисляет отложенную реферальную награду, когда приглашённый набрал
    REF_ACTIVATION_SPEND потраченных токенов. Вызывается после каждого списания."""
    ref_uid = ref_pending_award.get(uid)
    if ref_uid is None:
        return
    if lifetime_spent.get(uid, 0) < REF_ACTIVATION_SPEND:
        return

    ref_pending_award.pop(uid, None)
    referral_count[ref_uid] = referral_count.get(ref_uid, 0) + 1
    _grant_requests(ref_uid, REF_REWARD)
    save_state()
    try:
        await bot.send_message(
            ref_uid,
            f"✅ <b>Ваш приглашённый освоился в боте!</b>\n\n"
            f"➕ Начислено: <b>{fmt_tokens(REF_REWARD)}</b> токенов\n"
            f"👥 Всего активных приглашений: <b>{referral_count[ref_uid]}</b>",
        )
    except Exception:
        pass
    await _check_ref_milestones(bot, ref_uid)


async def _check_ref_milestones(bot: Bot, uid: int) -> None:
    """Разовые бонусы за реферальные уровни (см. REF_MILESTONES)."""
    count = referral_count.get(uid, 0)
    claimed = ref_milestone_claimed.get(uid, 0)
    for level in sorted(REF_MILESTONES):
        if count >= level > claimed:
            bonus = REF_MILESTONES[level]
            _grant_requests(uid, bonus)
            ref_milestone_claimed[uid] = level
            claimed = level
            save_state()
            try:
                await bot.send_message(
                    uid,
                    f"🏆 <b>Реферальный уровень: {level} друзей!</b>\n\n"
                    f"➕ Разовый бонус: <b>{fmt_tokens(bonus)}</b> токенов\n"
                    f"📊 Доступно сейчас: <b>{fmt_count(remaining(uid))}</b>",
                )
            except Exception:
                pass


def is_premium(uid: int) -> bool:
    """True, если у пользователя есть Premium-доступ (Premium-тариф или админ)."""
    return user_plan(uid).key != "free"


def ads_enabled(uid: int) -> bool:
    return user_plan(uid).show_ads


def category_locked_for(uid: int, cat_key: str) -> bool:
    """True, если категория премиальная, а у пользователя нет Premium-доступа."""
    return cat_key in PREMIUM_CATEGORIES and not is_premium(uid)


def fmt_count(n) -> str:
    """Форматирует лимит/остаток: бесконечность -> «∞»."""
    return "∞" if n == UNLIMITED else str(int(n))


def fmt_tokens(n) -> str:
    """Красивое число токенов: 36000 → «36 000», ∞ → «∞»."""
    if n == UNLIMITED:
        return "∞"
    return f"{int(n):,}".replace(",", " ")


# ── НАЗВАНИЯ КНОПОК ГЛАВНОГО МЕНЮ (Reply Keyboard) ─────────────
BTN_NEW_CHAT = "✨ Новый чат"
BTN_MODEL    = "🧠 Выбрать модель"
BTN_SETTINGS = "⚙️ Настройки"
BTN_BUY      = "💳 Пополнить баланс"
BTN_STATS    = "📊 Статистика"
BTN_HELP     = "❓ Помощь"
BTN_INVITE   = "🎁 Пригласить друга"
BTN_BONUS    = "🎯 Ежедневный бонус"

# ── ССЫЛКИ НА ДОКУМЕНТЫ (для согласования с банком) ───────────
USER_AGREEMENT_URL = "https://telegra.ph/Polzovatelskoe-soglashenie-04-01-19"
PRIVACY_POLICY_URL = "https://telegra.ph/Politika-konfidencialnosti-06-21-31"

# ── КОНТАКТ ПОДДЕРЖКИ (для согласования с банком) ─────────────
# Личный аккаунт-поддержка в Telegram. Группа не подходит — банк требует
# юзернейм / почту / тикет-систему.
SUPPORT_USERNAME = "@cuyodi"
SUPPORT_URL = "https://t.me/cuyodi"


# ══════════════════════════════════════════════════════════════
# МОДЕЛИ И КАТЕГОРИИ
# ══════════════════════════════════════════════════════════════
#
# Пользователь выбирает КАТЕГОРИЮ (не конкретную модель). Внутри категории
# бот сам перебирает модели по порядку при ошибке 429 (перегрузка) —
# переключение ВСЕГДА остаётся внутри границ выбранной категории.

@dataclass(frozen=True)
class ModelInfo:
    id: str            # ID модели у провайдера (OpenRouter или Featherless)
    name: str           # Отображаемое имя
    emoji: str          # Эмодзи модели
    desc: str           # Короткое описание
    knowledge: str      # Дата актуальности знаний
    nvidia: bool = False  # Требует согласия с уведомлением NVIDIA
    provider: str = "openrouter"  # "openrouter"|"featherless"|"freetheai"|"echogate"
    # Множитель списания токенов с баланса пользователя: дорогие модели
    # (например, через EchoGate) списывают фактические токены × коэффициент.
    token_coef: float = 1.0
    # Уровень рассуждений (EchoGate/OpenAI: none|low|medium|high|xhigh).
    # Пустая строка — параметр в запрос не передаётся.
    reasoning_effort: str = ""


@dataclass(frozen=True)
class ModelCategory:
    key: str                 # Внутренний ключ категории
    name: str                # Отображаемое имя категории
    emoji: str               # Эмодзи категории
    desc: str                # Описание того, для чего категория подходит
    models: tuple[str, ...]  # Ключи моделей в порядке предпочтения (fallback)
    # Системный промпт, специфичный для категории. Пустая строка -> берётся
    # общий SYSTEM_PROMPT. Дополняет базовые правила форматирования.
    system_extra: str = ""
    temperature: float = 0.7      # креативность генерации в этой категории
    max_tokens: int = MAX_TOKENS  # потолок длины ответа


MODELS: dict[str, ModelInfo] = {
    "deepseek_v4_flash": ModelInfo(
        id="deepseek/deepseek-v4-flash:free",
        name="DeepSeek V4 Flash",
        emoji="🐋",
        desc="Мощная универсальная модель: рассуждения, код, контекст 1М токенов.",
        knowledge="2026",
    ),
    "gpt_oss_20b": ModelInfo(
        id="openai/gpt-oss-20b:free",
        name="OpenAI: GPT-OSS 20B",
        emoji="🪶",
        desc="Компактная и быстрая версия GPT-OSS для простых задач.",
        knowledge="2025",
    ),
    "qwen3_coder_480b": ModelInfo(
        id="qwen/qwen3-coder:free",
        name="Qwen: Qwen3 Coder 480B A35B",
        emoji="💻",
        desc="Топ-модель для программирования: 1M контекст, лучший код.",
        knowledge="2025",
    ),
    "gemma4_31b": ModelInfo(
        id="google/gemma-4-31b-it:free",
        name="Google: Gemma 4 31B",
        emoji="⚖️",
        desc="Баланс общения, логики, программирования и зрения (vision).",
        knowledge="2025",
    ),
    "gemma4_26b": ModelInfo(
        id="google/gemma-4-26b-a4b-it:free",
        name="Google: Gemma 4 26B A4B",
        emoji="⚡",
        desc="Быстрая модель для общения и простых задач.",
        knowledge="2025",
    ),
    "nemotron3_super": ModelInfo(
        id="nvidia/nemotron-3-super-120b-a12b:free",
        name="NVIDIA: Nemotron 3 Super",
        emoji="🧠",
        desc="Сильна в логике, мультиагентных и технических задачах.",
        knowledge="2025",
        nvidia=True,
    ),
    "nemotron3_ultra": ModelInfo(
        id="nvidia/nemotron-3-ultra-550b-a55b:free",
        name="NVIDIA: Nemotron 3 Ultra",
        emoji="💻",
        desc="1M контекст, сильна в коде и сложных инструкциях.",
        knowledge="2025",
        nvidia=True,
    ),
    "nemotron_nano_9b": ModelInfo(
        id="nvidia/nemotron-nano-9b-v2:free",
        name="NVIDIA: Nemotron Nano 9B V2",
        emoji="🐭",
        desc="Маленькая и быстрая модель для простых задач и диалогов.",
        knowledge="2025",
        nvidia=True,
    ),
    "venice_uncensored": ModelInfo(
        id="cognitivecomputations/dolphin-mistral-24b-venice-edition:free",
        name="Venice: Uncensored",
        emoji="🎭",
        desc="Для свободного общения, творчества и персонажей.",
        knowledge="2025",
    ),
    "cohere_north_mini": ModelInfo(
        id="cohere/north-mini-code:free",
        name="Cohere: North Mini Code",
        emoji="💻",
        desc="Кодовый помощник: Python, исправление ошибок, агентные задачи.",
        knowledge="2025",
    ),
    "poolside_laguna_xs2": ModelInfo(
        id="poolside/laguna-xs.2:free",
        name="Poolside: Laguna XS.2",
        emoji="🧑‍💻",
        desc="Агентная модель для кода: генерация, рефакторинг, отладка.",
        knowledge="2026",
    ),
    "nemotron_nano_12b_vl": ModelInfo(
        id="nvidia/nemotron-nano-12b-v2-vl:free",
        name="NVIDIA: Nemotron Nano 12B 2 VL",
        emoji="👁",
        desc="Vision-Language: документы, видео, распознавание изображений.",
        knowledge="2025",
        nvidia=True,
    ),
    "nemotron_omni": ModelInfo(
        id="nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free",
        name="NVIDIA: Nemotron 3 Nano Omni",
        emoji="🖼️",
        desc="Мультимодальная: текст, изображение, видео.",
        knowledge="2025",
        nvidia=True,
    ),
    # ── Модели через FreeTheAi (provider="freetheai", отдельный ключ) ──
    "ftai_deepseek_pro": ModelInfo(
        id="olm/deepseek-v4-pro",
        name="DeepSeek: V4 Pro",
        emoji="🐋",
        desc="Сильная рассуждающая модель: математика, логика, код.",
        knowledge="2026",
        provider="freetheai",
    ),
    "ftai_gpt_54_mini": ModelInfo(
        id="bbl/gpt-5.4-mini",
        name="OpenAI: GPT-5.4 Mini",
        emoji="✨",
        desc="Компактный GPT-5.4: быстрый и универсальный.",
        knowledge="2026",
        provider="freetheai",
    ),
    "ftai_kimi_code": ModelInfo(
        id="olm/kimi-k2.7-code",
        name="Moonshot: Kimi K2.7 Code",
        emoji="💻",
        desc="Заточена под код: генерация, рефакторинг, отладка.",
        knowledge="2026",
        provider="freetheai",
    ),
    "ftai_minimax_m3": ModelInfo(
        id="min/minimax-m3",
        name="MiniMax: M3",
        emoji="🎭",
        desc="Длинный контекст, живые тексты и ролевые сценарии.",
        knowledge="2026",
        provider="freetheai",
    ),
    "ftai_mimo_vision": ModelInfo(
        id="mim/mimo-v2.5-pro",
        name="MiMo: v2.5 Pro (omni)",
        emoji="🖼️",
        desc="Мультимодальная omni-модель: текст + изображения.",
        knowledge="2026",
        provider="freetheai",
    ),
    # ── FreeTheAi: остальные согласованные модели ──
    "ftai_glm_52": ModelInfo(
        id="glm/glm-5.2",
        name="Z.ai: GLM-5.2",
        emoji="🚀",
        desc="Флагман GLM: 1M контекст, сильные рассуждения и код.",
        knowledge="2026",
        provider="freetheai",
    ),
    "ftai_deepseek_flash": ModelInfo(
        id="opc/deepseek-v4-flash-free",
        name="DeepSeek: V4 Flash",
        emoji="🐋",
        desc="Быстрая рассуждающая модель DeepSeek: логика, математика, код.",
        knowledge="2025",
        provider="freetheai",
    ),
    "ftai_gemini_35_flash": ModelInfo(
        id="bbl/gemini-3.5-flash",
        name="Google: Gemini 3.5 Flash",
        emoji="💎",
        desc="Быстрый Gemini: длинный контекст, универсальные задачи.",
        knowledge="2025",
        provider="freetheai",
    ),
    # ── Модели через EchoGate (provider="echogate", ключ forge-...) ──
    # Дорогая флагманская модель: списывает токены с баланса с коэффициентом ×3.
    "eg_gpt_56_luna": ModelInfo(
        id="gpt-5.6-luna",
        name="OpenAI: GPT-5.6 Luna",
        emoji="🌙",
        desc="Флагман GPT-5.6: топ-качество ответов. Списание токенов ×3.",
        knowledge="2026",
        provider="echogate",
        token_coef=3.0,
        reasoning_effort="none",   # без рассуждений: быстрее и дешевле
    ),
}

# Модели только для админов: скрыты из меню обычных пользователей, недоступны
# для явного выбора и исключаются из авто-ротации не-админов.
ADMIN_ONLY_MODEL_KEYS: set[str] = {"eg_gpt_56_luna"}

CATEGORIES: dict[str, ModelCategory] = {
    "coding": ModelCategory(
        key="coding",
        name="Кодинг",
        emoji="💻",
        desc="Для программирования, отладки, рефакторинга и работы с кодом.",
        models=(
            "deepseek_v4_flash",
            "qwen3_coder_480b",
            "nemotron3_ultra",
            "poolside_laguna_xs2",
            "cohere_north_mini",
            # FreeTheAi фолбэки (OpenRouter остаётся основным)
            "ftai_kimi_code",
            "ftai_deepseek_pro",
        ),
        system_extra=(
            "Ты — старший инженер-программист. Отвечай точно и по делу, без воды. "
            "Код давай в блоках с указанием языка. Если в вопросе есть ошибка — "
            "сначала кратко назови причину, затем исправленный код."
        ),
        temperature=0.2,
    ),
    "general": ModelCategory(
        key="general",
        name="Универсальные",
        emoji="🧠",
        desc="Для общения, анализа, текстов и большинства повседневных задач.",
        models=(
            "deepseek_v4_flash",
            "gemma4_31b",
            "nemotron3_super",
            # FreeTheAi фолбэки (OpenRouter остаётся основным)
            "ftai_glm_52",
            "ftai_deepseek_pro",
            "ftai_gpt_54_mini",
            "ftai_gemini_35_flash",
            # EchoGate (списание ×3) — в конце списка, чтобы в авто-режиме
            # использовалась лишь как последний фолбэк; обычно её выбирают явно.
            "eg_gpt_56_luna",
        ),
        temperature=0.7,
    ),
    "fast": ModelCategory(
        key="fast",
        name="Быстрые",
        emoji="⚡",
        desc="Компактные модели для быстрых ответов на простые вопросы.",
        models=(
            "gpt_oss_20b",
            "gemma4_26b",
            "nemotron_nano_9b",
            # FreeTheAi фолбэки (OpenRouter остаётся основным)
            "ftai_deepseek_flash",
            "ftai_gemini_35_flash",
            "ftai_gpt_54_mini",
        ),
        system_extra=(
            "Отвечай кратко и по существу — одним-двумя абзацами. "
            "Не растекайся, если пользователь не просит развёрнутый ответ."
        ),
        temperature=0.5,
    ),
    "creative": ModelCategory(
        key="creative",
        name="Творчество",
        emoji="🎭",
        desc="Свободное общение, ролевые персонажи и творческие тексты.",
        models=(
            "venice_uncensored",
            # FreeTheAi фолбэки (OpenRouter остаётся основным)
            "ftai_minimax_m3",
            "ftai_glm_52",
        ),
        system_extra=(
            "Ты — креативный собеседник и рассказчик. Пиши живо, образно, "
            "оставайся в роли, если пользователь задал персонажа или сценарий."
        ),
        temperature=0.9,
    ),
}

DEFAULT_CATEGORY = "general"

# Модели, реально умеющие «видеть» изображения (Vision-Language). Отдельной
# категории Vision больше нет: фото в ЛЮБОЙ категории автоматически уходит на
# первую доступную из этих моделей (см. ask_ai). Текстовые модели картинки не
# понимают, поэтому маршрутизируем только сюда.
VISION_MODEL_KEYS: tuple[str, ...] = (
    "gemma4_31b",
    "nemotron_nano_12b_vl",
    "nemotron_omni",
    "ftai_mimo_vision",   # FreeTheAi omni как фолбэк для зрения
)
VISION_TEMPERATURE = 0.4          # для анализа изображений креативность не нужна

NVIDIA_WARNING = (
    "⚠️ <b>NVIDIA предупреждение</b>\n\n"
    "Пожалуйста, не загружайте конфиденциальную информацию. "
    "Вы можете загружать изображения, которые мы и наши поставщики услуг "
    "будем использовать исключительно для предоставления вам возможности "
    "демонстрации. Ваше использование регистрируется в целях безопасности, "
    "а анонимные данные сеанса могут использоваться для улучшения продуктов "
    "и услуг NVIDIA, включая показатели производительности сети, результаты, "
    "сгенерированные ИИ, и расшифровки аудиозаписей. Зарегистрированные данные "
    "сеанса для целей улучшения не связаны с вашей личностью или каким-либо "
    "постоянным идентификатором."
)

SYSTEM_PROMPT = (
    "Ты — умный AI-ассистент в Telegram. "
    "Отвечай на языке пользователя (обычно русский). "
    "Будь полезным, точным и дружелюбным. "
    "Сообщения отображаются как Telegram Rich Message, поэтому используй Markdown: "
    "**жирный**, *курсив*, `код`, блоки кода с языком, заголовки (#, ##), "
    "нумерованные и маркированные списки, таблицы (| col | col |), цитаты (>). "
    "Любую математику — даже одну переменную или короткое выражение типа a^2 — "
    "ОБЯЗАТЕЛЬНО оборачивай в $...$ (строчная формула) или $$...$$ (блочная формула). "
    "Правильно: $(a+b)(a-b)=a^2-b^2$. "
    "Неправильно (НИКОГДА так не делай): (a+b)(a-b)=a^2-b^2 без знаков $ — "
    "обычные круглые скобки ( ) вокруг формулы НЕ заменяют $ и не превращают текст в формулу. "
    "НЕ используй \\[...\\], \\(...\\), \\boxed{} и другие LaTeX-обёртки — только $ и $$. "
    # Дата НЕ вшивается при старте: бот живёт неделями, замороженная дата
    # вводила модель в заблуждение. Подставляется на каждый запрос — см.
    # _system_prompt() ниже.
)


def _system_prompt() -> str:
    """SYSTEM_PROMPT со свежей датой на момент запроса."""
    return SYSTEM_PROMPT + (
        f"Текущая дата: {datetime.now(timezone.utc).strftime('%d.%m.%Y')}."
    )


# ══════════════════════════════════════════════════════════════
# ИНИЦИАЛИЗАЦИЯ
# ══════════════════════════════════════════════════════════════

router = Router(name="main")
dp = Dispatcher()
dp.include_router(router)


# Антиспам для алертов: не чаще одного сообщения об ошибке в 60 секунд,
# иначе циклическая ошибка зафлудит админа и упрётся в лимиты Telegram.
_last_error_alert_at: float = 0.0


@dp.errors()
async def on_unhandled_error(event: ErrorEvent) -> None:
    """Глобальный перехват необработанных исключений в любом хендлере.

    Логирует трейсбек, шлёт алерт админам (не чаще раза в 60с) и отвечает
    пользователю мягким сообщением, чтобы бот не выглядел «зависшим».
    """
    global _last_error_alert_at
    logging.error("Необработанная ошибка в хендлере", exc_info=event.exception)

    # Алерт админам (не чаще 1/60с, чтобы не зафлудить)
    now = time.monotonic()
    if now - _last_error_alert_at >= 60:
        _last_error_alert_at = now
        import traceback
        exc = event.exception
        tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
        upd = event.update
        who = ""
        try:
            u = (upd.message and upd.message.from_user) or \
                (upd.callback_query and upd.callback_query.from_user)
            if u:
                who = f" (от {u.id} @{u.username or '—'})"
        except Exception:
            pass
        # _mask_token обязателен: httpx кладёт URL запроса (а он содержит
        # BOT_TOKEN) в текст многих исключений, и трейсбек ушёл бы в чат как есть.
        # Маскируем ДО обрезки, иначе токен может остаться в хвосте.
        alert = f"🚨 Ошибка в боте{who}:\n<code>{html.quote(_mask_token(tb)[-2500:])}</code>"
        for admin_id in ADMIN_IDS:
            try:
                await upd.bot.send_message(admin_id, alert)
            except Exception:
                pass

    # Ответ пользователю
    upd = event.update
    msg = getattr(upd, "message", None)
    cb = getattr(upd, "callback_query", None)
    try:
        if cb is not None:
            await cb.answer("⚠️ Что-то пошло не так. Попробуйте ещё раз.", show_alert=True)
        elif msg is not None:
            await msg.answer("⚠️ Что-то пошло не так. Попробуйте ещё раз чуть позже.")
    except Exception:
        pass

http: Optional[httpx.AsyncClient] = None

# Кэш публичного статуса провайдеров: prefix -> объект из /v1/health.
# Он нужен, чтобы не обращаться к сайту при каждом открытии клавиатуры.
_model_health_cache: dict[str, dict] = {}
_model_health_cache_at: float = 0.0
_model_health_lock = asyncio.Lock()

histories: dict[int, list[dict]] = {}
usage: dict[int, dict] = {}

# Анти-флуд: uid'ы, для которых прямо сейчас генерируется ответ. Пока запрос
# в работе, новые сообщения этого пользователя отклоняются — иначе один юзер
# может параллельно запустить десятки запросов к моделям.
processing_users: set[int] = set()

# Время последнего запроса к AI по пользователю — для кулдауна USER_COOLDOWN_SEC.
# Только в памяти: на диск не пишется намеренно, после рестарта кулдаун сбрасывать
# не жалко. Чистится вместе с остальными словарями в _cleanup_loop.
last_ai_request_at: dict[int, float] = {}

# Глобальный семафор на исходящие запросы к моделям. Создаётся лениво, потому
# что на уровне модуля ещё нет работающего event loop.
_ai_semaphore: Optional[asyncio.Semaphore] = None


def _get_ai_semaphore() -> asyncio.Semaphore:
    global _ai_semaphore
    if _ai_semaphore is None:
        _ai_semaphore = asyncio.Semaphore(AI_MAX_CONCURRENT)
    return _ai_semaphore

# uid'ы, от которых бот ждёт текст-промпт для картинки (нажали "🎨 Нарисовать"
# или /image без текста). Следующее их сообщение уйдёт в генератор изображений.
# Индекс текущего HF-токена для ротации по дневной ZeroGPU-квоте.

user_categories: dict[int, str] = {}   # uid -> ключ категории (что выбрал юзер)
model_menu_source: dict[int, str] = {}  # uid -> откуда открыт выбор модели: "main" | "settings"
nvidia_consent: dict[int, bool] = {}
agreement_accepted: dict[int, bool] = {}  # uid -> принял пользовательское соглашение
captcha_solved: set[int] = set()          # uid -> прошёл робо-проверку (капчу)
user_stats: dict[int, dict] = {}
broadcast_lock = asyncio.Lock()
botohub_msg_counter: dict[int, int] = {}  # uid -> счётчик сообщений для рекламы
# uid -> уже видел приветку BotoHub. Не персистится сознательно (после рестарта
# приветка может показаться повторно — это не страшно), но и не чистится, поэтому
# ограничиваем размер: иначе набор растёт на каждого нового пользователя до
# перезапуска. Порядок вставки в set не определён, так что при переполнении
# просто очищаем — хуже повторной приветки ничего не случится.
BOTOHUB_KNOWN_LIMIT = 10_000
botohub_known_users: set[int] = set()
op_pass_date: dict[int, str] = {}         # uid -> ISO-дата последнего прохождения ОП (посуточный сброс)
op_free_used: dict[int, int] = {}         # uid -> сколько бесплатных ответов уже отдано
pending_ai: dict[int, dict] = {}          # uid -> отложенный запрос, ждущий прохождения ОП (in-memory)
pending_auto_switch: dict[int, dict] = {} # uid -> запрос, ожидающий Да/Нет после сбоя выбранной модели
user_msg_to_bot_msg: dict[int, dict[int, int]] = {}  # chat_id -> {user_msg_id: bot_msg_id}
# Сколько последних пар «сообщение юзера -> ответ бота» помним на чат. Нужно
# только для привязки reply/редактирования к свежим сообщениям, поэтому старые
# можно спокойно забывать — иначе внутренний dict рос бы бесконечно (утечка).
MSG_MAP_LIMIT = 200


def _remember_bot_msg(chat_id: int, user_msg_id: int, bot_msg_id: int) -> None:
    """Запоминает связь user_msg -> bot_msg, ограничивая размер истории на чат."""
    chat_map = user_msg_to_bot_msg.setdefault(chat_id, {})
    chat_map[user_msg_id] = bot_msg_id
    if len(chat_map) > MSG_MAP_LIMIT:
        # dict сохраняет порядок вставки — выкидываем самые старые записи.
        for old_key in list(chat_map)[: len(chat_map) - MSG_MAP_LIMIT]:
            chat_map.pop(old_key, None)
user_specific_model: dict[int, str] = {}  # uid -> model key (или None для авто)

# ── Реферальная система ────────────────────────────────────────
invited_by: dict[int, int] = {}      # uid новичка -> uid пригласившего (проставляется ПОСЛЕ выдачи награды = защита от повтора)
referral_count: dict[int, int] = {}  # uid -> сколько друзей он привёл (для статистики и наград)
pending_referral: dict[int, int] = {}  # uid новичка -> uid пригласившего ДО принятия соглашения (эфемерно, не persist)

# ── Персонализация ─────────────────────────────────────────────
user_verbosity: dict[int, str] = {}   # uid -> ключ длины ответа (short/medium/long)
user_persona: dict[int, str] = {}     # uid -> ключ персоны (см. PERSONAS)
user_code_files: dict[int, bool] = {} # uid -> отправлять fenced-код отдельными файлами (по умолчанию False)
daily_bonus_date: dict[int, str] = {} # uid -> ISO-дата последнего полученного ежедневного бонуса
daily_bonus_streak: dict[int, int] = {} # uid -> текущий стрик ежедневного бонуса (дней подряд)
ref_pending_award: dict[int, int] = {} # uid новичка -> uid пригласившего (награда ждёт активности новичка)
lifetime_spent: dict[int, int] = {} # uid -> потрачено токенов за всё время (для активации рефералов)

# ── Состояние новых фич: бан, продажи, промокоды, триал, скидка ──
banned_users: set[int] = set()             # заблокированные пользователи
banned_notified: set[int] = set()          # кому уже показали сообщение о бане
project_closed: bool = False               # проект закрыт (только админы могут работать)
project_closed_notified: set[int] = set()  # кому уже показали "проект закрыт"
purchases: list[dict] = []                 # журнал покупок: {ts, uid, kind, title, amount, currency}
promo_codes: dict[str, dict] = {}          # КОД -> {kind, value, max_uses, used, expires, users:[uid]}
trial_used: set[int] = set()               # кто уже активировал пробный Premium
# Кому уже выдали стартовые токены. Отдельная постоянная метка, а не проверка
# «uid нет в user_stats»: user_stats чистится для неактивных через 7 дней, и на
# старом условии подарок можно было получать заново каждую неделю (обнулить
# баланс → подождать → написать снова). В _cleanup_loop НЕ чистится.
welcome_granted: set[int] = set()
sale_info: dict = {}                       # глобальная скидка: {"percent": int, "until": iso-дата|None}
ref_milestone_claimed: dict[int, int] = {} # uid -> последний выданный реферальный уровень
user_custom_prompt: dict[int, str] = {}    # uid -> личная инструкция боту (Premium)
user_input_state: dict[int, str] = {}      # uid -> ожидание текстового ввода (свой промпт и т.п.)

# ── Платежи Platega ────────────────────────────────────────────
# tx_id -> {uid, kind, amount_rub, credited}. kind: "pack:<key>" | "premium:<days>".
# credited=True — награда уже начислена (защита от повторного зачисления при
# многократном нажатии «Я оплатил»). Хранится в SQLite, переживает рестарт.
platega_tx: dict[str, dict] = {}
# ── Платежи Telegram Stars: уже обработанные charge_id ─────────
# Telegram может доставить successful_payment повторно (например, если бот
# упал после начисления, но до подтверждения offset). Без леджера покупка
# зачислилась бы дважды. Хранится в SQLite, переживает рестарт.
# charge_id -> {uid, stars, payload, ts, refunded}. Раньше был просто set
# из charge_id; данные платежа нужны для возврата (refundStarPayment) из админки.
stars_charges: dict[str, dict] = {}
# tx_id, которые прямо сейчас проверяются в cb_pg_check. Внутрипамятный набор
# (НЕ сохраняется в SQLite) — защита от гонки при двойном «Я оплатил».
_platega_checking: set[str] = set()
# Защита от двойного клика «оплатить» при СОЗДАНИИ счёта (у проверки оплаты
# своя защита — _platega_checking). Без неё два быстрых клика = два счёта.
_platega_creating: set[int] = set()


# ════════════════════════════════════════════════════════════════
# PLATEGA — ПРИЁМ ОПЛАТЫ В РУБЛЯХ (API-клиент)
# ═══════════════════════════════════════════════════════════════════

def _platega_headers() -> dict:
    return {
        "X-MerchantId": PLATEGA_MERCHANT_ID,
        "X-Secret": PLATEGA_SECRET,
        "Content-Type": "application/json",
        "Accept": "application/json",
    }


async def _platega_create(uid: int, amount_rub: int, description: str,
                          payload: str, method_code: int,
                          user_name: str = "") -> Optional[dict]:
    """Создаёт транзакцию Platega (POST /transaction/process).

    Возвращает {"transactionId", "redirect", "status"} или None при ошибке.
    metadata.userId (Telegram ID) обязателен для антифрода Platega.
    """
    if not platega_enabled() or http is None:
        return None
    body = {
        "paymentMethod": method_code,
        "paymentDetails": {"amount": amount_rub, "currency": "RUB"},
        "description": description,
        "return": PLATEGA_RETURN_URL,
        "failedUrl": PLATEGA_FAILED_URL,
        "payload": payload,
        "metadata": {"userId": str(uid), "userName": user_name or str(uid)},
    }
    try:
        r = await http.post(
            f"{PLATEGA_BASE_URL}/transaction/process",
            headers=_platega_headers(), json=body, timeout=30,
        )
        if r.status_code != 200:
            logging.warning(f"Platega create {r.status_code}: {r.text[:300]}")
            return None
        data = r.json()
        tx_id = data.get("transactionId")
        pay_url = data.get("redirect") or data.get("url")
        if not tx_id or not pay_url:
            logging.warning(f"Platega create: неполный ответ {data}")
            return None
        return {"transactionId": tx_id, "redirect": pay_url, "status": data.get("status")}
    except Exception as e:
        logging.warning(f"Platega create error: {e}")
        return None


async def _platega_status(tx_id: str) -> Optional[dict]:
    """Возвращает ответ по транзакции (GET /transaction/{id}) как dict.

    Ключи: "status" (PENDING / CONFIRMED / CANCELED / CHARGEBACKED) и, если
    провайдер их отдаёт, сумма/валюта. None — при ошибке.
    """
    if not platega_enabled() or http is None:
        return None
    try:
        r = await http.get(
            f"{PLATEGA_BASE_URL}/transaction/{tx_id}",
            headers=_platega_headers(), timeout=20,
        )
        if r.status_code != 200:
            logging.warning(f"Platega status {r.status_code}: {r.text[:200]}")
            return None
        return r.json() or {}
    except Exception as e:
        logging.warning(f"Platega status error: {e}")
        return None


def _platega_paid_amount(data: dict) -> Optional[float]:
    """Достаёт фактически оплаченную сумму из ответа Platega, если она есть.

    Провайдер может отдавать сумму в разных местах/именах — проверяем самые
    вероятные. Возвращает None, если поле не найдено (тогда сверку пропускаем,
    чтобы не отклонять реальные оплаты из-за неизвестного формата)."""
    candidates = [
        data.get("amount"),
        data.get("paymentDetails", {}).get("amount") if isinstance(data.get("paymentDetails"), dict) else None,
        data.get("paidAmount"),
    ]
    for val in candidates:
        if val is None:
            continue
        try:
            return float(val)
        except (TypeError, ValueError):
            continue
    return None


async def _botohub_send(chat_id: int, hi: bool = False) -> None:
    """Отправляет рекламный пост через BotoHub Views API."""
    try:
        r = await http.post(
            BOTOHUB_URL,
            headers={"Authorization": BOTOHUB_TOKEN, "Content-Type": "application/json"},
            json={"SendToChatId": chat_id, "hi": hi},
            timeout=10,
        )
        data = r.json()
        code = data.get("SendPostResult", 0)
        if code != 1:
            logging.debug(f"BotoHub: chat_id={chat_id} hi={hi} result={code}")
    except Exception as e:
        logging.debug(f"BotoHub error: {e}")


async def _botohub_hi(uid: int) -> None:
    """Приветка для нового пользователя (раз в 24ч, только при /start)."""
    if uid not in botohub_known_users:
        if len(botohub_known_users) >= BOTOHUB_KNOWN_LIMIT:
            botohub_known_users.clear()
        botohub_known_users.add(uid)
        await _botohub_send(uid, hi=True)


async def _botohub_maybe_show_ad(uid: int) -> None:
    """Показывает рекламу каждые BOTOHUB_AD_EVERY сообщений."""
    if not ads_enabled(uid):
        return
    botohub_msg_counter[uid] = botohub_msg_counter.get(uid, 0) + 1
    if botohub_msg_counter[uid] % BOTOHUB_AD_EVERY == 0:
        await _botohub_send(uid, hi=False)


def user_category(uid: int) -> str:
    """Ключ выбранной пользователем категории (или дефолтная).

    Если ранее была выбрана премиальная категория, а Premium-доступа больше
    нет, — возвращаем дефолтную, чтобы не дать бесплатный доступ к платным
    моделям.
    """
    key = user_categories.get(uid, DEFAULT_CATEGORY)
    # Удалённая/неизвестная категория (например, старая «strongest» из БД) —
    # откатываемся на дефолтную, чтобы не ловить KeyError.
    if key not in CATEGORIES:
        return DEFAULT_CATEGORY
    if key in PREMIUM_CATEGORIES and not is_premium(uid):
        return DEFAULT_CATEGORY
    return key


# ── УПРАВЛЕНИЕ МОДЕЛЯМИ (админка): техработы / скрытие / добавление ──

# Кэш оверрайдов/описаний в памяти. Раньше КАЖДОЕ сообщение пользователя
# открывало 2+ новых sqlite-соединения и прогоняло весь CREATE TABLE-скрипт
# (через _db_connect), блокируя event loop. Данные меняются только из
# админки — кэш инвалидируется в _set_model_override/_set_model_description.
_model_overrides_cache: Optional[dict] = None
_model_desc_cache: dict[str, str] = {}


def _invalidate_model_caches() -> None:
    global _model_overrides_cache
    _model_overrides_cache = None
    _model_desc_cache.clear()


def _model_overrides() -> dict:
    """{model_key: {"state": ..., "data": {...}}} из SQLite (переживает перезапуск).

    state: "maintenance" (🔧 техработы — в ротации не участвует),
           "hidden" (скрыта), "custom" (добавлена через панель),
           "deleted" (удалена).
    """
    global _model_overrides_cache
    if _model_overrides_cache is not None:
        return _model_overrides_cache
    try:
        conn = _db_connect()
        rows = conn.execute(
            "SELECT model_key, state, data FROM model_overrides"
        ).fetchall()
        conn.close()
        out = {}
        for key, state, data in rows:
            try:
                payload = json.loads(data) if data else {}
            except Exception:
                payload = {}
            out[key] = {"state": state, "data": payload}
        _model_overrides_cache = out
        return out
    except Exception as e:
        logging.warning(f"model_overrides: ошибка чтения: {e}")
        return {}


def _set_model_override(key: str, state: Optional[str], data: Optional[dict] = None) -> None:
    """state=None снимает пометку/скрытие или удаляет добавленную модель."""
    try:
        conn = _db_connect()
        with conn:
            if state is None:
                conn.execute("DELETE FROM model_overrides WHERE model_key = ?", (key,))
            else:
                # Не стираем метаданные добавленной модели при переключении
                # техработ/скрытия/удаления.
                if data is None:
                    old = conn.execute(
                        "SELECT data FROM model_overrides WHERE model_key = ?", (key,)
                    ).fetchone()
                    if old and old[0]:
                        try:
                            data = json.loads(old[0])
                        except Exception:
                            data = {}
                conn.execute(
                    "INSERT INTO model_overrides(model_key, state, data) VALUES(?,?,?)"
                    " ON CONFLICT(model_key) DO UPDATE SET state=excluded.state, data=excluded.data",
                    (key, state, json.dumps(data or {}, ensure_ascii=False)),
                )
        conn.close()
        _invalidate_model_caches()
    except Exception as e:
        logging.warning(f"model_overrides: ошибка записи: {e}")


BUILTIN_MODEL_KEYS = frozenset(MODELS)


def _is_custom_override(key: str, row: Optional[dict] = None) -> bool:
    row = row or _model_overrides().get(key, {})
    return key not in BUILTIN_MODEL_KEYS and bool(row.get("data", {}).get("category"))


def _normal_model_state(key: str, row: Optional[dict] = None) -> Optional[str]:
    """Состояние модели после снятия временной пометки."""
    return "custom" if _is_custom_override(key, row) else None


def _model_description(key: str) -> str:
    """Описание из панели; если его не меняли — описание из ModelInfo."""
    if key in _model_desc_cache:
        return _model_desc_cache[key]
    try:
        conn = _db_connect()
        row = conn.execute(
            "SELECT description FROM model_descriptions WHERE model_key = ?", (key,)
        ).fetchone()
        conn.close()
        if row is not None:
            desc = row[0] or ""
            _model_desc_cache[key] = desc
            return desc
    except Exception as e:
        logging.warning("model_descriptions: ошибка чтения: %s", e)
    desc = MODELS[key].desc if key in MODELS else ""
    _model_desc_cache[key] = desc
    return desc


def _set_model_description(key: str, description: str) -> None:
    try:
        conn = _db_connect()
        with conn:
            conn.execute(
                "INSERT INTO model_descriptions(model_key, description) VALUES(?,?) "
                "ON CONFLICT(model_key) DO UPDATE SET description=excluded.description",
                (key, description),
            )
        conn.close()
        _invalidate_model_caches()
    except Exception as e:
        logging.warning("model_descriptions: ошибка записи: %s", e)


def _sync_custom_models() -> None:
    """Модели, добавленные через админку, подмешиваются в MODELS на лету."""
    overrides = _model_overrides()
    for key, row in overrides.items():
        if row["state"] == "deleted" and key not in BUILTIN_MODEL_KEYS:
            MODELS.pop(key, None)
            continue
        if not _is_custom_override(key, row) or key in MODELS:
            continue
        d = row["data"]
        try:
            MODELS[key] = ModelInfo(
                id=d.get("id", key),
                name=d.get("name", key),
                emoji=d.get("emoji", "🤖"),
                desc=d.get("desc", ""),
                knowledge=d.get("knowledge", "2025"),
                nvidia=False,
                provider=d.get("provider", "openrouter"),
                token_coef=float(d.get("token_coef", 1.0)),
            )
        except Exception:
            continue


def category_models(cat_key: str) -> tuple[str, ...]:
    """Список ключей моделей в категории, начиная с предпочитаемой.

    Скрытые (🗑) и стоящие на техработах (🔧) модели из ротации исключаются,
    добавленные через админку — дописываются в конец своей категории.
    """
    _sync_custom_models()
    overrides = _model_overrides()
    cat = CATEGORIES.get(cat_key, CATEGORIES[DEFAULT_CATEGORY])
    keys = [
        k for k in cat.models
        if overrides.get(k, {}).get("state") not in ("hidden", "maintenance", "deleted")
    ]
    keys += [
        k for k, row in overrides.items()
        if _is_custom_override(k, row)
        and row["state"] not in ("hidden", "maintenance", "deleted")
        and row["data"].get("category") == cat.key
    ]
    return tuple(dict.fromkeys(keys))


def visible_category_models(cat_key: str) -> tuple[str, ...]:
    """Модели, видимые в меню. Техработы видны, но выбрать их нельзя."""
    _sync_custom_models()
    overrides = _model_overrides()
    cat = CATEGORIES.get(cat_key, CATEGORIES[DEFAULT_CATEGORY])
    keys = [
        k for k in cat.models
        if overrides.get(k, {}).get("state") not in ("hidden", "deleted")
    ]
    keys += [
        k for k, row in overrides.items()
        if _is_custom_override(k, row)
        and row["state"] not in ("hidden", "deleted")
        and row["data"].get("category") == cat.key
    ]
    return tuple(k for k in dict.fromkeys(keys) if k in MODELS)


def _model_health_prefix(model_key: str) -> Optional[str]:
    """Префикс FreeTheAI из ID вида ``olm/deepseek-v4-pro``."""
    model = MODELS.get(model_key)
    if not model or model.provider != "freetheai" or "/" not in model.id:
        return None
    return model.id.split("/", 1)[0].lower()


def _model_health_view(model_key: str) -> tuple[str, Optional[float], str]:
    """(эмодзи, процент ошибок, подпись) по правилу 50%."""
    prefix = _model_health_prefix(model_key)
    if not prefix:
        return "", None, ""
    provider = _model_health_cache.get(prefix)
    if not provider:
        return "⚪", None, "нет данных"
    try:
        error_percent = max(0.0, min(100.0, float(provider.get("error_rate_30m", 0)) * 100))
    except (TypeError, ValueError):
        return "⚪", None, "нет данных"
    # Явный down всегда красный; в остальных случаях действует порог пользователя.
    is_bad = provider.get("status") == "down" or error_percent >= 50.0
    return ("🔴" if is_bad else "🟢"), error_percent, ("много ошибок" if is_bad else "работает")


async def _refresh_model_health(force: bool = False) -> bool:
    """Загружает статус с сайта. При ошибке сохраняет последний успешный кэш."""
    global _model_health_cache, _model_health_cache_at
    now = time.monotonic()
    if not force and _model_health_cache and now - _model_health_cache_at < MODEL_HEALTH_CACHE_TTL:
        return True
    async with _model_health_lock:
        now = time.monotonic()
        if not force and _model_health_cache and now - _model_health_cache_at < MODEL_HEALTH_CACHE_TTL:
            return True
        try:
            if http is None:
                async with httpx.AsyncClient(follow_redirects=True) as client:
                    response = await client.get(MODEL_HEALTH_API_URL, timeout=8)
            else:
                response = await http.get(MODEL_HEALTH_API_URL, timeout=8, follow_redirects=True)
            response.raise_for_status()
            data = response.json()
            providers = data.get("providers") if isinstance(data, dict) else None
            if not isinstance(providers, list):
                raise ValueError("health response has no providers list")
            fresh = {
                str(item.get("prefix", "")).lower(): item
                for item in providers
                if isinstance(item, dict) and item.get("prefix")
            }
            if not fresh:
                raise ValueError("health providers list is empty")
            _model_health_cache = fresh
            _model_health_cache_at = time.monotonic()
            return True
        except Exception as exc:
            logging.warning("Model health fetch failed: %s", exc)
            return bool(_model_health_cache)


def _model_health_text() -> str:
    """Короткий список FreeTheAI-моделей для сообщения в Telegram."""
    lines = [
        "🩺 <b>Доступность моделей</b>",
        "",
        "🟢 — ошибок меньше 50%",
        "🔴 — ошибок 50% или больше",
        "⚪ — свежие данные не получены",
        "",
    ]
    seen: set[tuple[str, str]] = set()
    for model_key, model in MODELS.items():
        prefix = _model_health_prefix(model_key)
        if not prefix:
            continue
        identity = (prefix, model.id)
        if identity in seen:
            continue
        seen.add(identity)
        icon, error_percent, label = _model_health_view(model_key)
        if error_percent is None:
            lines.append(f"{icon} <b>{html.quote(model.name)}</b> — {label}")
        else:
            pretty = f"{error_percent:.1f}".rstrip("0").rstrip(".")
            lines.append(
                f"{icon} <b>{html.quote(model.name)}</b> — {pretty}% ошибок · {label}"
            )
    lines.extend(["", "<i>Показатели за последние 30 минут.</i>"])
    return "\n".join(lines)


async def _ping_provider(name: str, url: str) -> str:
    """Строка «пинг до провайдера»: меряет время лёгкого GET к его API.

    Ответ 4xx (401/405 и т.п.) — это норм: сервер жив и ответил, нас
    интересует только время и что это не 5xx/таймаут.
    """
    started = time.monotonic()
    try:
        r = await http.get(url, timeout=8)
        ms = int((time.monotonic() - started) * 1000)
        if r.status_code >= 500:
            return f"🔴 <b>{name}</b> — ошибка {r.status_code} · {ms} мс"
        return f"🟢 <b>{name}</b> — {ms} мс"
    except Exception:
        ms = int((time.monotonic() - started) * 1000)
        return f"🔴 <b>{name}</b> — не отвечает ({ms} мс)"


async def _provider_ping_text() -> str:
    """Блок «Пинг до провайдеров» для сообщения о доступности моделей."""
    # Названия провайдеров пользователям не показываем — нейтральные подписи.
    targets = [("Основной сервер", OPENROUTER_URL)]
    if FEATHERLESS_KEY:
        targets.append(("Резервный сервер", FEATHERLESS_URL))
    if FREETHEAI_KEY:
        targets.append(("Дополнительный сервер", FREETHEAI_URL))
    lines = await asyncio.gather(*(_ping_provider(n, u) for n, u in targets))
    return "📡 <b>Пинг до провайдеров</b>\n\n" + "\n".join(lines)


async def _health_message_text() -> str:
    """Полное сообщение «Доступность моделей»: пинг провайдеров + статусы."""
    return await _provider_ping_text() + "\n\n" + _model_health_text()


def _model_health_kb() -> InlineKeyboardMarkup:
    b = InlineKeyboardBuilder()
    b.row(InlineKeyboardButton(text="🌐 Подробный статус на сайте", url=MODEL_HEALTH_PAGE_URL))
    b.row(InlineKeyboardButton(text="🔄 Обновить список", callback_data="health:refresh"))
    b.row(InlineKeyboardButton(text="⬅️ В меню", callback_data="menu:main"))
    return b.as_markup()


# ══════════════════════════════════════════════════════════════
# TYPING ACTION
# ══════════════════════════════════════════════════════════════

class TypingIndicator:
    """Удобный контекст-менеджер для индикатора 'печатает...' в чате."""

    def __init__(self, bot: Bot, chat_id: int) -> None:
        self._bot = bot
        self._chat_id = chat_id
        self._stop = asyncio.Event()
        self._task: Optional[asyncio.Task] = None

    async def __aenter__(self) -> "TypingIndicator":
        self._task = asyncio.create_task(self._loop())
        return self

    async def __aexit__(self, *exc_info) -> None:
        self._stop.set()
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except (asyncio.CancelledError, Exception):
                pass

    async def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                await self._bot.send_chat_action(chat_id=self._chat_id, action="typing")
            except Exception:
                pass
            try:
                await asyncio.wait_for(asyncio.shield(self._stop.wait()), timeout=4.0)
            except asyncio.TimeoutError:
                pass


# ═══════════════════════════════════════════════════════════════
# ЛИМИТЫ
# ══════════════════════════════════════════════════════════════

def _uid(msg: Message) -> int:
    return msg.from_user.id if msg.from_user else msg.chat.id


def _get_usage(uid: int) -> dict:
    today = date.today()
    u = usage.setdefault(uid, {"date": today, "used": 0, "bought": 0})
    if u["date"] != today:
        u["date"] = today
        u["used"] = 0
    return u


def can_use(uid: int) -> bool:
    """Хватает ли токенов хотя бы на один ответ (админы — всегда).

    Проверяем не «> 0», а минимальную стоимость ответа: иначе пользователь
    с 1 токеном получал полный ответ ценой MIN_TOKENS_SPEND фактически бесплатно.
    """
    if uid in ADMIN_IDS:
        return True
    return _get_usage(uid)["bought"] >= MIN_TOKENS_SPEND


def remaining(uid: int):
    """Текущий баланс токенов (∞ у админов)."""
    if uid in ADMIN_IDS:
        return UNLIMITED
    return int(_get_usage(uid)["bought"])


def _estimate_tokens(text) -> int:
    """Грубая оценка числа токенов по длине текста (≈3 символа на токен)."""
    if not isinstance(text, str):
        text = str(text or "")
    return max(1, len(text) // CHARS_PER_TOKEN_EST)


def _estimate_tokens_from_messages(messages) -> int:
    """Оценка input-токенов: системный промпт + история + вопрос."""
    total = 0
    for m in messages:
        c = m.get("content", "")
        if isinstance(c, str):
            total += len(c)
        elif isinstance(c, list):  # мультимодальный контент (текст + фото)
            for part in c:
                if isinstance(part, dict) and part.get("type") == "text":
                    total += len(part.get("text", ""))
    return max(1, total // CHARS_PER_TOKEN_EST)


# Оценка input-токенов последнего запроса (заполняется в ask_ai)
last_input_tokens: dict[int, int] = {}
# Коэффициент списания модели, давшей последний ответ (token_coef из ModelInfo).
# Заполняется в ask_ai при успехе; списание читает и очищает его через pop.
last_token_coef: dict[int, float] = {}


def _calc_spend(uid: int, tin: int, tout: int, coef: float = 1.0) -> int:
    """Сколько токенов списать за ответ (Premium — на 20% дешевле).
    coef — множитель модели (token_coef): дорогие модели списывают ×N.
    Для админов сумма СЧИТАЕТСЯ (показывается в квитанции для проверки),
    но с баланса не снимается — это решает место списания в _run_ai_turn."""
    spent = max(MIN_TOKENS_SPEND, int((int(tin) + int(tout)) * max(coef, 0.0)))
    if premium_active(uid):
        spent = int(spent * PREMIUM_SPEND_DISCOUNT)
    return spent


def _spend_tokens(uid: int, amount: int) -> None:
    """Списывает токены с баланса; used копит потраченное за день (статистика)."""
    u = _get_usage(uid)
    amt = max(0, int(amount))
    u["bought"] = max(0, int(u["bought"]) - amt)
    u["used"] = int(u["used"]) + amt
    lifetime_spent[uid] = lifetime_spent.get(uid, 0) + amt


# ══════════════════════════════════════════════════════════════
# BOTOHUB — ОБЯЗАТЕЛЬНАЯ ПОДПИСКА (ОП, продвинутая интеграция)
# ══════════════════════════════════════════════════════════════
# Продвинутая интеграция BotoHub: POST botohub.me/get-tasks с {"chat_id": ...}.
# BotoHub возвращает {"tasks": [ссылки], "completed", "skip"}. Пост со
# спонсорами и кнопкой «✅ Проверить» (callback_data=bh_check) формируем и
# отправляем МЫ САМИ по списку ссылок.
#   completed=True — подписался на всех → пропускаем к согласию
#   skip=True      — спонсоров нет → пропускаем без ОП
#   tasks пуст     — показывать нечего → тоже пропускаем
#   иначе          — показываем ссылки и ждём подписки
# Порядок онбординга нового пользователя: /start → ОП → (подписался ИЛИ
# купил Premium) → экран согласия → бот.


def _today_iso() -> str:
    """Сегодняшняя дата в ISO (для посуточного сброса ОП)."""
    return date.today().isoformat()


def _op_required(uid: int) -> bool:
    """Нужен ли пользователю ОП-гейт прямо сейчас.

    Логика: новичку первые OP_FREE_ANSWERS ответов бесплатны (чтобы зацепить),
    затем ОП требуется раз в день. Прошёл ОП сегодня — весь день без ОП; на
    следующий день (новые спонсоры у BotoHub) ОП снова. Админы и Premium
    не видят ОП никогда.
    """
    if uid in ADMIN_IDS or premium_active(uid):
        return False
    if op_free_used.get(uid, 0) < OP_FREE_ANSWERS:   # ещё есть бесплатные ответы
        return False
    return op_pass_date.get(uid) != _today_iso()      # сегодня ОП ещё не пройден


async def _op_check(uid: int) -> dict:
    """Дёргает продвинутую интеграцию BotoHub (botohub.me/get-tasks).

    Возвращает {"tasks": list[str], "completed": bool, "skip": bool}.
    При отсутствии токена / любой ошибке / не-200 — fail-open:
    {"tasks": [], "completed": False, "skip": True}, чтобы недоступность
    BotoHub НИКОГДА не блокировала пользователей.
    """
    if not BOTOHUB_OP_TOKEN or http is None:
        return {"tasks": [], "completed": False, "skip": True}
    try:
        r = await http.post(
            BOTOHUB_OP_URL,
            headers={"Auth": BOTOHUB_OP_TOKEN, "Content-Type": "application/json"},
            json={"chat_id": uid},
            timeout=15,
        )
        if r.status_code != 200:
            logging.warning(f"BotoHub OP {r.status_code}: {r.text[:200]}")
            return {"tasks": [], "completed": False, "skip": True}
        data = r.json()
        tasks = [str(t) for t in (data.get("tasks") or []) if t]
        return {
            "tasks": tasks,
            "completed": bool(data.get("completed")),
            "skip": bool(data.get("skip")),
        }
    except Exception as e:
        logging.warning(f"BotoHub OP error: {e}")
        return {"tasks": [], "completed": False, "skip": True}


def _op_tasks_kb(tasks: list[str]) -> InlineKeyboardBuilder:
    """Пост ОП, который формируем сами: кнопки-ссылки на спонсоров,
    кнопка «✅ Проверить» (bh_check) и наши доп-кнопки (отказ / премиум)."""
    b = InlineKeyboardBuilder()
    for i, url in enumerate(tasks, 1):
        b.row(InlineKeyboardButton(text=f"📢 Спонсор {i}", url=url))
    b.row(InlineKeyboardButton(text="✅ Проверить", callback_data="bh_check"))
    b.row(InlineKeyboardButton(text="😕 Не хочу подписываться", callback_data="op_nope"))
    return b


def _op_premium_kb() -> InlineKeyboardBuilder:
    """Оффер Premium на ОП-гейте. Здесь ТОЛЬКО Premium (он снимает ОП);
    пакеты токенов не показываем — они не снимают подписку, иначе человек
    заплатит, но останется за шлагбаумом. Цены — с учётом активной скидки;
    при подключённой Platega доступна оплата рублями."""
    b = InlineKeyboardBuilder()
    if platega_enabled():
        b.row(InlineKeyboardButton(
            text=f"💎 Premium на {PREMIUM_DAYS} дней — {_disc(PREMIUM_PRICE)}⭐ / {_disc(PREMIUM_PRICE_RUB)}₽",
            callback_data="pmenu:premium",
        ))
    else:
        b.row(InlineKeyboardButton(
            text=f"💎 Premium на {PREMIUM_DAYS} дней — {_disc(PREMIUM_PRICE)} ⭐",
            callback_data="buy_premium",
        ))
    b.row(InlineKeyboardButton(text="⬅️ Закрыть", callback_data="op_back"))
    return b


_OP_WAIT_TEXT = (
    "🤝 <b>Чтобы продолжить, подпишитесь на спонсоров ниже 👇</b>\n\n"
    "После подписки на все каналы нажмите «✅ Проверить».\n\n"
    "Не хотите подписываться? Оформите 💎 Premium — тогда обязательная "
    "подписка и реклама не нужны."
)


async def _run_op_gate(msg: Message, uid: int) -> bool:
    """Прогоняет ОП через продвинутую интеграцию BotoHub. True — пользователь
    прошёл на сегодня (completed/skip/нет спонсоров). False — ждём подписки:
    сами шлём пост со ссылками на спонсоров, кнопкой «Проверить» и своими
    кнопками (отказаться / купить премиум)."""
    res = await _op_check(uid)
    tasks = res.get("tasks") or []
    if res["completed"] or res["skip"] or not tasks:
        op_pass_date[uid] = _today_iso()
        save_state()
        return True
    await msg.answer(_OP_WAIT_TEXT, reply_markup=_op_tasks_kb(tasks).as_markup())
    return False


# ══════════════════════════════════════════════════════════════
# OPENROUTER — CHAT AI
# ══════════════════════════════════════════════════════════════════

def _strip_code_and_formulas(text: str) -> str:
    """Вырезает код и формулы перед эвристиками «мусора».

    В коде и LaTeX легитимны и повторы (----, ====, ....), и длинные
    «слова» без пробелов (LaTeX-команды, цепочки вызовов, пути) —
    оценивать их эвристиками деградации нельзя (ложные «сбои»
    на длинных ответах reasoning-моделей типа NVIDIA Nemotron).
    """
    t = re.sub(r"```.*?(?:```|$)", " ", text, flags=re.DOTALL)  # блоки кода
    t = re.sub(r"`[^`\n]+`", " ", t)                            # инлайн-код
    t = re.sub(r"\$\$.*?\$\$", " ", t, flags=re.DOTALL)         # формулы $$...$$
    t = re.sub(r"\$[^$\n]+\$", " ", t)                          # формулы $...$
    return t


def _looks_like_gibberish(text: str) -> bool:
    """Эвристика: похоже ли на ВЫРОЖДЕННЫЙ (мусорный) ответ модели.

    Бесплатные модели иногда «зацикливаются» и выдают поток повторяющихся или
    склеенных токенов (напр. «urpurpurp Logo Logo ...»). Такой ответ надо
    отбросить и попробовать следующую модель, а не показывать пользователю.

    Проверки намеренно консервативны, чтобы НЕ забраковать нормальный текст
    или код: короткие ответы не оцениваются вовсе.
    """
    if not text or not text.strip():
        return True
    # Код и формулы не оцениваем: там повторы и длинные токены — норма.
    t = _strip_code_and_formulas(text).strip()
    if not t:
        return False  # ответ целиком из кода/формул — это валидный ответ

    # 1) Одна короткая подстрока (2–5 симв.) повторяется подряд 5+ раз:
    #    "urpurpurpurp", "<unk><unk><unk>..." — мусор, даже в коротком тексте.
    #    НО только если в повторе есть буквы/цифры: повторы пунктуации
    #    (-----, =====, ....., │││) — легитимное оформление таблиц и
    #    разделителей в длинных ответах, а не деградация модели.
    if len(t) >= 20:
        for m in re.finditer(r"(.{2,5}?)\1{4,}", t):
            if re.search(r"[^\W_]", m.group(1)):
                return True

    if len(t) < 40:
        return False  # слишком короткий, чтобы судить (напр. «42», «Готово»)

    words = re.findall(r"\S+", t)

    # 2) Зацикливание на словах. Считаем частоты и смотрим, не «забивают» ли
    #    одно-два слова весь ответ (типичный признак деградации модели).
    if len(words) >= 8:
        freqs = Counter(w.lower() for w in words)
        top = freqs.most_common(2)
        top_word, top_n = top[0]
        # Слова без букв/цифр (|, ---, • и т.п.) — оформление, не зацикливание.
        has_alnum = bool(re.search(r"[^\W_]", top_word))
        # 2a) одно слово (длиннее 2 симв.) занимает >30% текста
        if has_alnum and len(top_word) >= 3 and top_n / len(words) > 0.30:
            return True
        # 2b) два самых частых слова вместе занимают >40% текста
        top2_n = sum(n for _, n in top)
        if has_alnum and len(top) == 2 and len(top_word) >= 3 and top2_n / len(words) > 0.40:
            return True

    # 3) Экстремально длинный «токен» без пробелов — склеенные токены.
    #    Исключаем ссылки/пути/base64, а остальное бракуем только при
    #    явной периодичности/монотонности внутри — иначе под раздачу
    #    попадали длинные легитимные конструкции (LaTeX вне $...$ и т.п.).
    for w in words:
        if len(w) <= 45 or re.match(r"https?://|[A-Za-z0-9+/=_.\-]+$", w):
            continue
        if re.search(r"(.{1,6})\1{3,}", w) or len(set(w)) <= 5:
            return True

    return False


async def _call_model(model_id: str, messages: list, *, temperature: float = 0.7,
                       max_tokens: int = MAX_TOKENS, on_delta=None,
                       provider: str = "openrouter",
                       reasoning_effort: str = "",
                       timeout: float | None = None,
                       strip_think: bool = True) -> str:
    """Запрос к провайдеру моделей со стримингом (stream=True).

    provider выбирает эндпоинт и ключ: "openrouter" (по умолчанию) или
    "featherless" (OpenAI-совместимый API, см. FEATHERLESS_URL/KEY).
    По мере генерации вызывает on_delta(накопленный_текст) — вызывающий сам
    решает, как часто реально обновлять сообщение (троттлинг). Возвращает
    полный текст ответа. Ошибки статуса (429/401/...) поднимаются как
    httpx.HTTPStatusError с уже прочитанным телом.
    """
    payload = {
        "model": model_id,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": True,
    }
    # Уровень рассуждений (reasoning-модели EchoGate/OpenAI): "none" полностью
    # отключает размышления. Не задан — параметр не отправляем вовсе.
    if reasoning_effort:
        payload["reasoning_effort"] = reasoning_effort
    if provider == "featherless":
        url = FEATHERLESS_URL
        api_key = FEATHERLESS_KEY
    elif provider == "freetheai":
        url = FREETHEAI_URL
        api_key = FREETHEAI_KEY
    elif provider == "echogate":
        url = ECHOGATE_URL
        api_key = ECHOGATE_KEY
    else:
        url = OPENROUTER_URL
        api_key = OPENROUTER_KEY
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        # Атрибуция приложения — рекомендована и OpenRouter, и Featherless.
        "HTTP-Referer": "https://t.me/your_bot",
        "X-Title": "Telegram AI Bot",
    }
    # Временные сбои шлюза (502/503/504 — часто HTML-заглушка Cloudflare, а не
    # JSON API) бывают мгновенными и общими для всех моделей: без ретрая бот за
    # секунду впустую перебирал бы весь список. Делаем до 3 попыток одной модели
    # с короткой паузой (учитываем Retry-After, если пришёл).
    GATEWAY_RETRY_CODES = {502, 503, 504}
    MAX_ATTEMPTS = 3
    parts: list[str] = []
    reasoning_parts: list[str] = []
    # Семафор держим на весь запрос, включая ретраи: он ограничивает не число
    # вызовов, а число ОДНОВРЕМЕННЫХ соединений к провайдерам — иначе общие
    # ключи бесплатных тарифов выгорают по частоте. acquire/release вместо
    # async with, чтобы не переставлять отступы всего тела цикла.
    _sem = _get_ai_semaphore()
    await _sem.acquire()
    try:
      for attempt in range(MAX_ATTEMPTS):
        parts.clear()
        reasoning_parts.clear()
        async with http.stream(
            "POST", url, json=payload, headers=headers,
            timeout=timeout if timeout is not None else REQUEST_TIMEOUT,
        ) as r:
            if r.status_code != 200:
                # Тело нужно прочитать до raise, иначе e.response.text недоступен.
                await r.aread()
                if r.status_code in GATEWAY_RETRY_CODES and attempt < MAX_ATTEMPTS - 1:
                    try:
                        wait = float(r.headers.get("Retry-After", "2"))
                    except (TypeError, ValueError):
                        wait = 2.0
                    wait = min(max(wait, 1.0), 8.0)  # держим в разумных рамках
                    logging.warning(
                        "Шлюз %s для %s (попытка %d/%d), ждём %.0fс и повторяем",
                        r.status_code, model_id, attempt + 1, MAX_ATTEMPTS, wait,
                    )
                    await asyncio.sleep(wait)
                    continue
                r.raise_for_status()
            # ДИАГНОСТИКА: копим сырые непустые строки ответа, чтобы при пустом
            # результате увидеть, что реально прислал провайдер (ошибка checkin,
            # другое поле, не-SSE тело и т.п.). Ограничиваем объём — только начало.
            raw_debug = []
            async for line in r.aiter_lines():
                if line and line.strip() and len(raw_debug) < 15:
                    raw_debug.append(line[:300])
                if not line or not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    obj = json.loads(data)
                except Exception:
                    continue
                # Провайдер может прислать ошибку прямо в теле стрима (HTTP 200),
                # напр. {"error":{"message":"provider temporarily unavailable"}}.
                # Ловим её явно и поднимаем наверх — иначе ответ выглядел бы «пустым».
                err = obj.get("error")
                if err:
                    if isinstance(err, dict):
                        emsg = err.get("message") or "unknown provider error"
                        etype = err.get("type") or "provider_error"
                    else:
                        emsg, etype = str(err), "provider_error"
                    # Пробрасываем и type, и message: type — для ветвления, message —
                    # для показа пользователю. Разделитель "|" разбираем в ask_ai.
                    raise RuntimeError(f"stream_error:{etype}|{emsg}")
                choice = (obj.get("choices") or [{}])[0]
                delta = choice.get("delta") or {}
                piece = delta.get("content")
                # Reasoning-модели (GLM и др.) шлют размышления отдельным полем.
                # Копим его на случай, если content останется пустым.
                rpiece = delta.get("reasoning_content") or delta.get("reasoning")
                if rpiece:
                    reasoning_parts.append(rpiece)
                if piece:
                    parts.append(piece)
                    if on_delta is not None:
                        await on_delta("".join(parts))
        # Успешно дочитали стрим (HTTP 200) — больше не повторяем.
        break
    finally:
        _sem.release()

    answer = "".join(parts)
    # Некоторые модели кладут весь ответ в reasoning, а content оставляют пустым —
    # тогда показываем reasoning, иначе пользователь получил бы «пустой» ответ.
    if not answer.strip() and reasoning_parts:
        answer = "".join(reasoning_parts)
    # Вырезаем служебные блоки размышлений <think>...</think>, если модель
    # вписала их прямо в content. strip_think=False — когда ответ может
    # ЛЕГАЛЬНО содержать такие подстроки (например, self-edit возвращает
    # исходник бота, где этот же regex записан буквально).
    if strip_think:
        answer = re.sub(r"<think>.*?</think>", "", answer, flags=re.DOTALL).strip()
    else:
        answer = answer.strip()
    # ДИАГНОСТИКА: если ответ пустой — покажем сырьё, чтобы понять причину.
    if not answer:
        logging.warning(
            "ПУСТОЙ ответ [provider=%s model=%s]. Сырые строки ответа:\n%s",
            provider, model_id,
            "\n".join(raw_debug) if raw_debug else "<тело пустое / не в формате data:>",
        )
    return answer


def _rollback(uid: int) -> None:
    """Убирает последнее сообщение пользователя из истории при полном провале.

    Ассистента здесь снимать НЕ нужно, хотя выглядит так, будто нужно: между
    histories[uid].append({"role": "assistant", ...}) и return ans в ask_ai нет
    ни одного await, а отмена (в т.ч. по AI_TURN_TIMEOUT) доставляется только в
    точке await. Проверено запуском: когда wait_for бросает TimeoutError, запись
    ассистента ещё не выполнена, а если корутина успела дойти до конца — wait_for
    возвращает результат, а не таймаут. Добавить сюда pop() ассистента означало бы
    при следующем сбое удалять реальный предыдущий обмен.
    """
    h = histories.get(uid, [])
    if h and h[-1]["role"] == "user":
        h.pop()


class SelectedModelUnavailable(RuntimeError):
    """Конкретно выбранная модель не ответила; нужен выбор пользователя."""

    def __init__(self, model_key: str, reason: str = "unavailable") -> None:
        self.model_key = model_key
        self.reason = reason
        super().__init__(reason)


class AIError(str):
    """Текст ошибки от ask_ai. Ведёт себя как обычная строка, но отличим по типу.

    Нужен потому, что «это ошибка?» раньше определялось по первому символу
    ответа (_ERROR_PREFIXES). Первый символ выбирает МОДЕЛЬ, то есть в конечном
    счёте пользователь: достаточно попросить «начинай ответ с ⚠️», и каждый ответ
    считался ошибкой — токены не списывались, бесплатный ответ не сгорал, а текст
    приходил целиком. Тип подделать через промпт нельзя.
    """

    __slots__ = ()


async def ask_ai(uid: int, content, status_msg: Optional[Message] = None,
                  category: Optional[str] = None) -> str:
    """
    Отправляет запрос к моделям и стримит ответ в status_msg (живое превью).

    Для текста бот перебирает модели СТРОГО внутри выбранной категории. Если
    content — это список (изображение), запрос уходит на vision-модели
    (VISION_MODEL_KEYS) НЕЗАВИСИМО от категории: только они умеют «видеть» фото.
    """
    is_image = isinstance(content, list)
    specific: Optional[str] = None
    strict_specific = False

    if is_image:
        # Фото понимают только vision-модели — маршрутизируем на них.
        model_keys = [k for k in VISION_MODEL_KEYS if k in MODELS]
        cat_key = "vision"
        gen_temp = VISION_TEMPERATURE
        gen_max = MAX_TOKENS
        sys_prompt = _system_prompt()
    else:
        cat_key = category or user_category(uid)
        cat = CATEGORIES.get(cat_key, CATEGORIES[DEFAULT_CATEGORY])
        cat_models = list(category_models(cat_key))
        specific = user_specific_model.get(uid)
        strict_specific = bool(specific and specific in cat_models)
        if strict_specific:
            # Явный выбор больше не ведёт себя как Auto: сначала спрашиваем
            # разрешение пользователя, а не молча перебираем остальные модели.
            model_keys = [specific]
        else:
            model_keys = cat_models
        # Админ-модели (например, EchoGate GPT-5.6 Luna) — только владельцу:
        # у остальных выпадают и из авто-ротации, и из явного выбора.
        if not _is_admin(uid):
            model_keys = [k for k in model_keys if k not in ADMIN_ONLY_MODEL_KEYS]
            if not model_keys:  # выбранная модель оказалась админской → Auto
                model_keys = [k for k in category_models(cat_key)
                              if k not in ADMIN_ONLY_MODEL_KEYS]
        gen_temp = cat.temperature
        # Длина ответа (verbosity): потолок токенов не выше лимита категории.
        verb = user_verbosity_key(uid)
        gen_max = min(cat.max_tokens, VERBOSITY_TOKENS[verb])
        # Общий промпт + добавка категории (кодинг/творчество/быстрые) +
        # персона пользователя + подсказка по длине ответа.
        extras = [cat.system_extra, user_persona_obj(uid).system_extra, VERBOSITY_HINT[verb]]
        if is_premium(uid) and user_custom_prompt.get(uid):
            extras.append(f"Дополнительные пожелания пользователя: {user_custom_prompt[uid]}")
        sys_prompt = _system_prompt()
        for ex in extras:
            if ex:
                sys_prompt += f"\n\n{ex}"

    hist = histories.setdefault(uid, [])
    hist.append({"role": "user", "content": content})
    if len(hist) > MAX_HISTORY:
        histories[uid] = hist[-MAX_HISTORY:]

    messages = [{"role": "system", "content": sys_prompt}] + histories[uid]
    last_input_tokens[uid] = _estimate_tokens_from_messages(messages)

    # Живое превью стриминга: редактируем status_msg простым текстом не чаще
    # раза в ~1.2с (иначе упрёмся в лимиты Telegram на редактирование).
    stream_state = {"last": 0.0, "shown": ""}

    async def _on_delta(acc: str) -> None:
        if not status_msg:
            return
        now = time.monotonic()
        if now - stream_state["last"] < 1.2 or not acc.strip():
            return
        stream_state["last"] = now
        preview = acc[-3500:]
        if preview == stream_state["shown"]:
            return
        stream_state["shown"] = preview
        try:
            await status_msg.edit_text(preview + " ▌", parse_mode=None)
        except Exception:
            pass

    last_error = ""
    last_provider_error = ""
    for i, key in enumerate(model_keys):
        # .get, а не [key]: строка стоит ВНЕ try, и KeyError здесь убивал весь
        # ответ вместо перехода к следующей модели (юзер получал «что-то пошло
        # не так» и висящий статус). Ключ пропадает, если модель удалили в
        # админке или список категории разошёлся с MODELS после перезагрузки.
        model = MODELS.get(key)
        if model is None:
            logging.warning(f"модель {key} отсутствует в MODELS — пропускаю")
            last_error = "no_model"
            continue
        stream_state["last"] = 0.0  # сбрасываем троттл на каждую новую модель

        try:
            ans = await _call_model(
                model.id, messages, temperature=gen_temp,
                max_tokens=gen_max, on_delta=_on_delta,
                provider=model.provider,
                reasoning_effort=model.reasoning_effort,
            )

            # Модель могла «зациклиться» и выдать мусор (или пустоту) —
            # не показываем это пользователю, пробуем следующую модель.
            if _looks_like_gibberish(ans):
                last_error = "bad_output"
                logging.warning(
                    f"Мусорный/пустой ответ от {key} (категория {cat_key}), "
                    f"пробуем следующую модель"
                )
                next_model = MODELS.get(model_keys[i + 1]) if i < len(model_keys) - 1 else None
                if status_msg and next_model is not None:
                    try:
                        await status_msg.edit_text(
                            f"⚠️ <i>Модель <b>{html.quote(model.name)}</b> дала сбой, "
                            f"пробуем <b>{html.quote(next_model.name)}</b>...</i>"
                        )
                    except Exception:
                        pass
                continue

            histories[uid].append({"role": "assistant", "content": ans})
            # Запоминаем множитель списания ответившей модели (для ×N-моделей).
            last_token_coef[uid] = model.token_coef
            # Финальный красивый ответ отправит _run_ai_turn (Rich Message);
            # здесь дополнительно ничего не редактируем.
            return ans

        except httpx.TimeoutException:
            last_error = "timeout"
            logging.warning(f"Таймаут модели {key} (категория {cat_key})")
            continue

        except httpx.HTTPStatusError as e:
            status_code = e.response.status_code
            body = e.response.text
            logging.warning(f"{model.provider} {status_code} для {key}: {body[:200]}")

            # Машиночитаемый тип ошибки из тела (FreeTheAi/OpenAI-envelope).
            etype = ""
            try:
                etype = ((e.response.json() or {}).get("error") or {}).get("type") or ""
            except Exception:
                pass

            # 403 daily_checkin_required — ключ жив, но владелец не сделал
            # /checkin сегодня. Дальше перебирать бессмысленно: у FreeTheAi
            # ляжет ВСЯ группа. Сообщаем явно и выходим.
            if status_code == 403 and (
                etype == "daily_checkin_required" or "daily_checkin_required" in body
            ):
                _rollback(uid)
                if strict_specific and specific:
                    raise SelectedModelUnavailable(specific, "daily_checkin_required")
                return AIError(
                    "🔧 <b>Часть моделей временно недоступна.</b>\n\n"
                    "Попробуйте другую модель или повторите попытку немного позже."
                )

            if status_code == 429:
                last_error = "429"
                next_model = MODELS.get(model_keys[i + 1]) if i < len(model_keys) - 1 else None
                if status_msg and next_model is not None:
                    try:
                        await status_msg.edit_text(
                            f"⚠️ <i>Модель <b>{html.quote(model.name)}</b> ограничена, "
                            f"пробуем <b>{html.quote(next_model.name)}</b> из той же категории...</i>"
                        )
                    except Exception:
                        pass
                continue
            elif status_code == 401:
                _rollback(uid)
                if strict_specific and specific:
                    raise SelectedModelUnavailable(specific, "unauthorized")
                return AIError("🔧 Сервис модели временно недоступен. Попробуйте немного позже.")
            else:
                # 502/503/504 и прочее — провайдер сбоит, пробуем следующую модель.
                last_error = "provider"
                last_provider_error = etype or f"HTTP {status_code}"
                continue

        except Exception as e:
            emsg = str(e)
            # Ошибка провайдера прямо из стрима: формат "stream_error:<type>|<msg>".
            if emsg.startswith("stream_error:"):
                payload = emsg[len("stream_error:"):]
                etype, _, human = payload.partition("|")
                if etype == "daily_checkin_required":
                    _rollback(uid)
                    if strict_specific and specific:
                        raise SelectedModelUnavailable(specific, "daily_checkin_required")
                    return AIError(
                        "🔧 <b>Часть моделей временно недоступна.</b>\n\n"
                        "Попробуйте другую модель или повторите попытку немного позже."
                    )
                last_error = "provider"
                last_provider_error = human or etype or "provider error"
            else:
                last_error = emsg
            logging.error(f"AI error ({key}): {e}")
            continue

    _rollback(uid)
    if strict_specific and specific:
        reason = last_provider_error or last_error or "unavailable"
        raise SelectedModelUnavailable(specific, reason)
    if last_error == "429":
        return AIError("😔 Все модели этой категории сейчас перегружены. Попробуйте через минуту или выберите другую категорию (/model).")
    if last_error == "timeout":
        return AIError("⏱️ Таймаут. Попробуйте ещё раз.")
    if last_error == "provider":
        return AIError(
            "😔 Провайдер этих моделей сейчас недоступен "
            f"(<code>{last_provider_error}</code>).\n\n"
            "Это временный сбой на стороне сервиса, а не бота. "
            "Попробуйте через пару минут или выберите другую категорию (/model)."
        )
    if last_error == "bad_output":
        return AIError("😔 Модели сейчас отвечают нестабильно. Попробуйте переспросить или сменить категорию (/model).")
    return AIError("❌ Не удалось получить ответ. Попробуйте позже.")


# ══════════════════════════════════════════════════════════════
# MISTRAL VOXTRAL — STT
# ══════════════════════════════════════════════════════════════

async def transcribe(audio_bytes: bytes, filename: str = "audio.ogg") -> str:
    try:
        r = await http.post(
            MISTRAL_STT_URL,
            headers={"Authorization": f"Bearer {MISTRAL_KEY}"},
            files={"file": (filename, audio_bytes)},
            data={"model": MISTRAL_STT_MODEL},
            timeout=60,
        )
        r.raise_for_status()
        return r.json().get("text", "")
    except Exception as e:
        logging.error(f"STT error: {e}")
        return ""


# ══════════════════════════════════════════════════════════════
# RICH MESSAGES (Bot API 10.1)
# ═════════════════════════════════════════════════════════════════

TELEGRAM_API_URL = f"https://api.telegram.org/bot{BOT_TOKEN}"


def _mask_token(text: str) -> str:
    """Убирает BOT_TOKEN из строки перед логированием: httpx включает URL
    запроса (с токеном) в текст многих исключений."""
    return str(text).replace(BOT_TOKEN, "***")


async def _send_rich_message(chat_id: int, rich_message: dict, reply_to: Optional[int] = None) -> Optional[int]:
    payload = {"chat_id": chat_id, "rich_message": rich_message}
    if reply_to:
        payload["reply_parameters"] = {"message_id": reply_to}
    try:
        r = await http.post(f"{TELEGRAM_API_URL}/sendRichMessage", json=payload, timeout=30)
        data = r.json()
        if data.get("ok"):
            return data["result"]["message_id"]
        logging.warning(f"sendRichMessage отклонён: {data.get('description')}")
        return None
    except Exception as e:
        logging.warning(f"sendRichMessage ошибка сети: {_mask_token(e)}")
        return None


_MATH_MARKER = re.compile(
    r'\^\{?-?\d|_\{|\\times|\\div|\\frac|\\sqrt|\\cdot|\\bullet|\\pm|\\neq|\\leq|\\geq|\\approx'
    r'|\\to|\\rightarrow|\\leftarrow|\\sum|\\prod|\\int|\\partial|\\infty|\\alpha|\\beta|\\pi'
)
_LOOKS_LIKE_MATH_CHARS = re.compile(r'^[\sa-zA-Z0-9+\-*/=^_{}.,\\()]+$')
_WORDY = re.compile(r'[a-zA-Zа-яА-ЯёЁ]{3,}')

_LATEX_SYMBOL_MAP = {
    "cdot": "•", "bullet": "•", "times": "×", "div": "÷", "pm": "±", "mp": "∓",
    "leq": "≤", "geq": "≥", "neq": "≠", "approx": "≈", "infty": "∞", "sqrt": "√",
    "to": "→", "rightarrow": "→", "leftarrow": "←", "leftrightarrow": "↔",
    "Rightarrow": "⇒", "Leftarrow": "⇐", "Leftrightarrow": "⇔",
    "alpha": "α", "beta": "β", "gamma": "γ", "Gamma": "Γ", "delta": "δ", "Delta": "Δ",
    "epsilon": "ε", "varepsilon": "ε", "zeta": "ζ", "eta": "η", "theta": "θ",
    "vartheta": "ϑ", "iota": "ι", "kappa": "κ", "lambda": "λ", "Lambda": "Λ",
    "mu": "μ", "nu": "ν", "xi": "ξ", "pi": "π", "rho": "ρ", "sigma": "σ",
    "Sigma": "Σ", "tau": "τ", "upsilon": "υ", "phi": "φ", "varphi": "φ", "Phi": "Φ",
    "chi": "χ", "psi": "ψ", "Psi": "Ψ", "omega": "ω", "Omega": "Ω",
    "sum": "∑", "prod": "∏", "int": "∫", "iint": "∬", "iiint": "∭", "oint": "∮",
    "partial": "∂", "nabla": "∇", "in": "∈", "notin": "∉", "ni": "∋",
    "subset": "⊂", "subseteq": "⊆", "supset": "⊃", "supseteq": "⊇",
    "cup": "∪", "cap": "∩", "setminus": "∖", "emptyset": "∅", "varnothing": "∅",
    "forall": "∀", "exists": "∃", "nexists": "∄", "equiv": "≡", "propto": "∝",
    "sim": "∼", "simeq": "≃", "cong": "≅", "perp": "⊥", "parallel": "∥",
    "angle": "∠", "degree": "°", "circ": "∘", "vee": "∨", "wedge": "∧",
    "lnot": "¬", "neg": "¬", "oplus": "⊕", "otimes": "⊗",
    "ldots": "…", "cdots": "⋯", "vdots": "⋮", "ddots": "⋱",
    "aleph": "ℵ", "hbar": "ℏ", "ell": "ℓ", "Re": "ℜ", "Im": "ℑ",
    "top": "⊤", "bot": "⊥",
}

# Сортируем имена команд по длине (длинные сначала), чтобы при сборке
# regex-альтернативы более длинная команда (например "infty") всегда
# проверялась раньше своего префикса ("in") и не "съедалась" по частям.
_LATEX_CMD_PATTERN = re.compile(
    r'\\(' + '|'.join(sorted(map(re.escape, _LATEX_SYMBOL_MAP), key=len, reverse=True)) + r')(?![a-zA-Z])'
)


def _latex_to_symbols(expr: str) -> str:
    """Заменяет LaTeX-команды (\\cdot, \\bullet, \\alpha, ...) на юникод-символы."""
    return _LATEX_CMD_PATTERN.sub(lambda m: _LATEX_SYMBOL_MAP[m.group(1)], expr)


def _looks_like_bare_formula(inner: str) -> bool:
    if not inner or '$' in inner:
        return False
    if not _LOOKS_LIKE_MATH_CHARS.match(inner):
        return False
    without_latex_cmds = re.sub(r'\\[a-zA-Z]+', '', inner)
    if _WORDY.search(without_latex_cmds):
        return False
    return bool(
        _MATH_MARKER.search(inner)
        or re.search(r'[a-zA-Z0-9]\s*=\s*[a-zA-Z0-9]', inner)
    )


def _wrap_bare_formulas(text: str) -> str:
    result = []
    i, n = 0, len(text)
    while i < n:
        if text[i] != '(':
            result.append(text[i])
            i += 1
            continue
        depth = 1
        j = i + 1
        aborted = False
        while j < n and depth > 0:
            ch = text[j]
            if ch == '(':
                depth += 1
            elif ch == ')':
                depth -= 1
            elif ch == '$' or ch == '\n':
                aborted = True
                break
            j += 1
        if aborted or depth != 0:
            # Незакрытая скобка (или внутри встретили $ / перенос строки).
            # Сохраняем весь прочитанный кусок как есть, чтобы не потерять
            # содержимое скобки (раньше после этого оставался только символ '(').
            result.append(text[i:j])
            i = j
            continue
        segment = text[i:j]
        inner_full = text[i + 1:j - 1]
        if _looks_like_bare_formula(inner_full):
            result.append(f'${segment}$')
        else:
            result.append('(')
            result.append(_wrap_bare_formulas(inner_full))
            result.append(')')
        i = j
    return ''.join(result)


def _normalize_latex(text: str) -> str:
    # \[...\] -> $$...$$
    text = re.sub(r'\\\[\s*(.*?)\s*\\\]', lambda m: f'$${m.group(1)}$$', text, flags=re.DOTALL)

    # Голые [ ... ] на своих строках -> $$...$$  (модели часто так пишут)
    text = re.sub(
        r'(?m)^\[\s*\n(.*?)\n\s*\]$',
        lambda m: f'$$\n{m.group(1)}\n$$',
        text,
        flags=re.DOTALL,
    )

    # \(...\) -> $...$
    text = re.sub(r'\\\(\s*(.*?)\s*\\\)', lambda m: f'${m.group(1)}$', text, flags=re.DOTALL)
    # \boxed{...} -> содержимое
    text = re.sub(r'\\boxed\{([^}]+)\}', r'\1', text)

    # Голые формулы в скобках
    text = _wrap_bare_formulas(text)

    # LaTeX-команды (\cdot, \alpha, \to, ...) -> юникод-символы.
    # Применяем ко всему тексту одним проходом — и внутри $...$/$$...$$,
    # и за пределами формул (модели иногда пишут \cdot прямо в обычном тексте).
    text = _latex_to_symbols(text)

    return text


def _rich_text(s: str) -> list[dict]:
    """Конвертирует inline-Markdown в список RichText-объектов (Bot API 10.1)."""
    result = []
    pattern = re.compile(
        r'(`[^`]+`)'
        r'|(\$\$[^$]+?\$\$)'
        r'|(\$[^$\n]+?\$)'
        r'|(\*\*[^*]+?\*\*)'
        r'|(\*[^*\n]+?\*)'
        r'|(~~[^~\n]+?~~)'
        r'|(\|\|[^|]+?\|\|)'
        r'|(\+\+[^+\n]+?\+\+)'
        r'|(__[^_\n]+?__)'
        r'|(_[^_\n]+?_)'
        r'|(\[([^\]]+)\]\((https?://[^)]+)\))'
    )
    pos = 0
    for m in pattern.finditer(s):
        if m.start() > pos:
            result.append({"type": "plain", "text": s[pos:m.start()]})
        raw = m.group(0)
        if raw.startswith('`'):
            result.append({"type": "code", "text": raw[1:-1]})
        elif raw.startswith('$$'):
            result.append({"type": "mathematical_expression", "expression": raw[2:-2].strip()})
        elif raw.startswith('$'):
            result.append({"type": "mathematical_expression", "expression": raw[1:-1].strip()})
        elif raw.startswith('**'):
            result.append({"type": "bold", "text": _rich_text(raw[2:-2])})
        elif raw.startswith('~~'):
            result.append({"type": "strikethrough", "text": _rich_text(raw[2:-2])})
        elif raw.startswith('||'):
            result.append({"type": "spoiler", "text": _rich_text(raw[2:-2])})
        elif raw.startswith('++'):
            result.append({"type": "underline", "text": _rich_text(raw[2:-2])})
        elif raw.startswith('__') and raw.endswith('__'):
            result.append({"type": "underline", "text": _rich_text(raw[2:-2])})
        elif raw.startswith('*') or raw.startswith('_'):
            result.append({"type": "italic", "text": _rich_text(raw[1:-1])})
        elif raw.startswith('['):
            result.append({"type": "link", "text": _rich_text(m.group(12)), "url": m.group(13)})
        pos = m.end()
    if pos < len(s):
        result.append({"type": "plain", "text": s[pos:]})
    return result if result else [{"type": "plain", "text": s}]


def _markdown_to_input_rich_message(text: str) -> Optional[dict]:
    try:
        text = _normalize_latex(text)
        blocks = []
        lines = text.split('\n')
        i = 0

        def _parse_list_items(start_i: int, bullet_re, sub_re=None) -> tuple[list, int]:
            items = []
            ci = start_i
            while ci < len(lines):
                line = lines[ci]
                m = bullet_re.match(line)
                if not m:
                    break
                item_text = bullet_re.sub('', line, count=1).strip()
                ci += 1
                sub_items = []
                if sub_re and ci < len(lines):
                    while ci < len(lines) and re.match(r'^\s{2,}', lines[ci]):
                        stripped = lines[ci].strip()
                        sub_m = sub_re.match(stripped)
                        if sub_m:
                            sub_items.append({"text": _rich_text(sub_re.sub('', stripped, count=1).strip())})
                        ci += 1
                entry: dict = {"text": _rich_text(item_text)}
                if sub_items:
                    entry["items"] = sub_items
                items.append(entry)
            return items, ci

        _ordered_re = re.compile(r'^\s*\d+[.)]\s+')
        _unordered_re = re.compile(r'^\s*[-*+]\s+')

        while i < len(lines):
            line = lines[i]

            if not line.strip():
                i += 1
                continue

            # Блок кода
            if line.strip().startswith('```'):
                lang = line.strip()[3:].strip()
                code_lines = []
                i += 1
                while i < len(lines) and not lines[i].strip().startswith('```'):
                    code_lines.append(lines[i])
                    i += 1
                i += 1
                blocks.append({
                    "type": "preformatted",
                    "text": "\n".join(code_lines),
                    **({"language": lang} if lang else {}),
                })
                continue

            # Блочная формула $$
            stripped = line.strip()
            if stripped.startswith('$$'):
                rest = stripped[2:]
                if '$$' in rest:
                    expr = rest[:rest.rfind('$$')]
                    blocks.append({"type": "mathematical_expression", "expression": expr.strip()})
                    i += 1
                    continue
                expr_lines = [rest]
                i += 1
                while i < len(lines) and '$$' not in lines[i]:
                    expr_lines.append(lines[i])
                    i += 1
                if i < len(lines):
                    expr_lines.append(lines[i][:lines[i].find('$$')])
                    i += 1
                blocks.append({
                    "type": "mathematical_expression",
                    "expression": "\n".join(expr_lines).strip(),
                })
                continue

            # Спойлер-блок >!
            if line.startswith('>!'):
                spoiler_lines = []
                while i < len(lines) and lines[i].startswith('>!'):
                    spoiler_lines.append(lines[i][2:].rstrip('!<').strip())
                    i += 1
                blocks.append({"type": "spoiler", "text": _rich_text(" ".join(spoiler_lines))})
                continue

            # Цитата >
            if line.startswith('>'):
                quote_lines = []
                while i < len(lines) and lines[i].startswith('>'):
                    quote_lines.append(lines[i].lstrip('> ').lstrip('>'))
                    i += 1
                blocks.append({"type": "block_quotation", "text": _rich_text("\n".join(quote_lines))})
                continue

            # Горизонтальная линия
            if re.match(r'^\s*[-*_]{3,}\s*$', line):
                blocks.append({"type": "divider"})
                i += 1
                continue

            # Заголовок
            heading_m = re.match(r'^(#{1,6})\s+(.+)$', line)
            if heading_m:
                level = len(heading_m.group(1))
                blocks.append({
                    "type": "section_heading",
                    "text": _rich_text(heading_m.group(2)),
                    "level": level,
                })
                i += 1
                continue

            # Нумерованный список
            if _ordered_re.match(line):
                items, i = _parse_list_items(i, _ordered_re, _unordered_re)
                blocks.append({"type": "list", "style": "ordered", "items": items})
                continue

            # Маркированный список
            if _unordered_re.match(line):
                items, i = _parse_list_items(i, _unordered_re, _unordered_re)
                blocks.append({"type": "list", "style": "unordered", "items": items})
                continue

            # Таблица
            if '|' in line and line.strip().startswith('|'):
                table_lines = []
                while i < len(lines) and '|' in lines[i] and lines[i].strip().startswith('|'):
                    if not re.match(r'^\s*\|[-:\s|]+\|\s*$', lines[i]):
                        table_lines.append(lines[i])
                    i += 1
                if table_lines:
                    rows = []
                    header_done = False
                    for tl in table_lines:
                        cells = [c.strip() for c in tl.strip('|').split('|')]
                        row: dict = {"cells": [{"text": _rich_text(c)} for c in cells]}
                        if not header_done:
                            row["is_header"] = True
                            header_done = True
                        rows.append(row)
                    blocks.append({"type": "table", "rows": rows})
                continue

            # Абзац
            _is_block_start = lambda ln: any([
                ln.startswith('#'), ln.startswith('>'), ln.startswith('```'),
                ln.startswith('$$'), _ordered_re.match(ln), _unordered_re.match(ln),
                re.match(r'^\s*[-*_]{3,}\s*$', ln),
                '|' in ln and ln.strip().startswith('|'),
            ])
            para_lines = [line]
            i += 1
            while i < len(lines) and lines[i].strip() and not _is_block_start(lines[i]):
                para_lines.append(lines[i])
                i += 1
            blocks.append({"type": "paragraph", "text": _rich_text(' '.join(para_lines))})

        if not blocks:
            return None
        return {"blocks": blocks}

    except Exception as e:
        logging.warning(f"_markdown_to_input_rich_message ошибка: {e}, текст: {text[:80]!r}")
        return None


def _markdown_to_rich_chunks(text: str) -> Optional[list[dict]]:
    if not text or not text.strip():
        return None

    # LaTeX-команды (\cdot -> •, \alpha -> α и т.д.) нормализуем ДО передачи
    # в любой из движков рендеринга, чтобы они применялись независимо от того,
    # какой путь (telegramify-markdown или собственный конвертер) сработает.
    text = _normalize_latex(text)

    if TELEGRAMIFY_AVAILABLE:
        try:
            items = telegramify_markdown.telegramify_rich(text)
            result = [item.to_dict() for item in items]
            if result:
                return result
        except Exception as e:
            logging.debug(f"telegramify_rich не смог: {e}")

    rich_msg = _markdown_to_input_rich_message(text)
    if not rich_msg or not rich_msg.get("blocks"):
        return None

    all_blocks = rich_msg["blocks"]
    chunks = []
    for start in range(0, len(all_blocks), RICH_MAX_BLOCKS):
        chunk_blocks = all_blocks[start:start + RICH_MAX_BLOCKS]
        if chunk_blocks:
            chunks.append({"blocks": chunk_blocks})
    return chunks if chunks else None


def _md_to_html(text: str) -> str:
    text = _normalize_latex(text)

    def replace_code_block(m):
        code = html.quote(m.group(2))
        return f"<pre><code>{code}</code></pre>"
    text = re.sub(r'```(\w*)\n?(.*?)```', replace_code_block, text, flags=re.DOTALL)

    text = re.sub(r'`([^`]+)`', lambda m: f"<code>{html.quote(m.group(1))}</code>", text)
    text = re.sub(r'\$\$(.+?)\$\$', lambda m: f"<pre><code>{m.group(1).strip()}</code></pre>", text, flags=re.DOTALL)
    text = re.sub(r'\$([^\$\n]+?)\$', lambda m: f"<code>{m.group(1).strip()}</code>", text)
    text = re.sub(r'^\s*[-*_]{3,}\s*$', '──────────', text, flags=re.MULTILINE)
    text = re.sub(r'^#{1,6}\s+(.+)$', lambda m: f"<b>{m.group(1)}</b>", text, flags=re.MULTILINE)
    text = re.sub(r'\*\*(.+?)\*\*', lambda m: f"<b>{m.group(1)}</b>", text, flags=re.DOTALL)
    text = re.sub(r'~~(.+?)~~', lambda m: f"<s>{m.group(1)}</s>", text, flags=re.DOTALL)
    text = re.sub(r'\+\+(.+?)\+\+', lambda m: f"<u>{m.group(1)}</u>", text, flags=re.DOTALL)
    text = re.sub(r'__(.+?)__', lambda m: f"<u>{m.group(1)}</u>", text, flags=re.DOTALL)
    text = re.sub(r'\|\|(.+?)\|\|', lambda m: f'<tg-spoiler>{m.group(1)}</tg-spoiler>', text, flags=re.DOTALL)
    text = re.sub(r'\[([^\]]+)\]\((https?://[^)]+)\)', lambda m: f'<a href="{html.quote(m.group(2))}">{html.quote(m.group(1))}</a>', text)
    text = re.sub(r'(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)', lambda m: f"<i>{m.group(1)}</i>", text)
    text = re.sub(r'(?<!_)_(?!_)(.+?)(?<!_)_(?!_)', lambda m: f"<i>{m.group(1)}</i>", text)

    def replace_blockquote(m):
        lines_bq = m.group(0).splitlines()
        content = "\n".join(l.lstrip("> ").lstrip(">") for l in lines_bq)
        return f"<blockquote>{content}</blockquote>"
    text = re.sub(r'(^>[ \t]?.+$(\n^>[ \t]?.+$)*)', replace_blockquote, text, flags=re.MULTILINE)

    return text


async def _send_as_file(msg: Message, text: str) -> Optional[int]:
    """Отправляет длинный ответ .md-файлом + короткая подпись-превью."""
    try:
        doc = BufferedInputFile(text.encode("utf-8"), filename="ответ.md")
        preview = text[:250].strip()
        sent = await msg.answer_document(
            doc,
            caption=(
                "📄 Ответ получился большим — прислал файлом.\n\n"
                f"<i>{html.quote(preview)}…</i>"
            ),
        )
        return sent.message_id
    except Exception as e:
        logging.error(f"send_as_file error: {e}")
        return None


async def _send_text_response(msg: Message, text: str) -> Optional[int]:
    """Отправляет обычную текстовую часть ответа с текущим Rich/fallback-пайплайном."""
    if not text:
        return None

    # Очень длинный ответ отдаём файлом, а не десятком сообщений — но только
    # для обычных ответов (сообщения об ошибках короткие и до порога не дойдут).
    if len(text) > SEND_AS_FILE_THRESHOLD and not _is_error_answer(text):
        file_id = await _send_as_file(msg, text)
        if file_id is not None:
            return file_id
        # если файл не ушёл — падаем в обычную отправку сообщениями ниже

    chat_id = msg.chat.id
    reply_to = msg.message_id
    first_id: Optional[int] = None

    rich_chunks = _markdown_to_rich_chunks(text)
    if rich_chunks:
        all_ok = True
        for i, chunk in enumerate(rich_chunks):
            chunk_bytes = len(json.dumps(chunk, ensure_ascii=False).encode())
            if chunk_bytes > RICH_MAX_BYTES:
                half = len(chunk["blocks"]) // 2 or 1
                sub_chunks = [
                    {"blocks": chunk["blocks"][:half]},
                    {"blocks": chunk["blocks"][half:]},
                ]
                for sub in sub_chunks:
                    if not sub["blocks"]:
                        continue
                    mid = await _send_rich_message(chat_id, sub, reply_to=reply_to if i == 0 else None)
                    if mid is None:
                        all_ok = False
                        break
                    if first_id is None:
                        first_id = mid
                    await asyncio.sleep(0.3)
                if not all_ok:
                    break
            else:
                mid = await _send_rich_message(chat_id, chunk, reply_to=reply_to if i == 0 else None)
                if mid is None:
                    all_ok = False
                    break
                if first_id is None:
                    first_id = mid
            if i < len(rich_chunks) - 1:
                await asyncio.sleep(0.3)
        if all_ok:
            return first_id
        try:
            await msg.answer("⚠️ <i>Rich-формат недоступен, ответ продолжен plain-текстом.</i>")
        except Exception:
            pass

    html_text = _md_to_html(text)
    chunks = _split(html_text, MAX_TG)
    for i, chunk in enumerate(chunks):
        try:
            sent = await msg.answer(chunk, parse_mode=ParseMode.HTML)
            if first_id is None:
                first_id = sent.message_id
        except Exception:
            try:
                sent = await msg.answer(chunk, parse_mode=None)
                if first_id is None:
                    first_id = sent.message_id
            except Exception as e:
                logging.error(f"send_response plain fallback error: {e}")
        if i < len(chunks) - 1:
            await asyncio.sleep(0.3)
    return first_id


_CODE_FENCE_RE = re.compile(r"```([^`\r\n]*)\r?\n(.*?)```", re.DOTALL)
_CODE_EXTENSIONS = {
    "python": "py", "py": "py", "javascript": "js", "js": "js",
    "typescript": "ts", "ts": "ts", "html": "html", "css": "css",
    "json": "json", "sql": "sql", "bash": "sh", "shell": "sh",
    "sh": "sh", "java": "java", "cpp": "cpp", "c++": "cpp",
    "c": "c", "csharp": "cs", "cs": "cs", "go": "go", "rust": "rs",
    "php": "php", "kotlin": "kt", "swift": "swift", "yaml": "yaml",
    "yml": "yml", "xml": "xml", "markdown": "md", "md": "md",
    "dockerfile": "Dockerfile", "text": "txt", "txt": "txt",
}
_DEFAULT_CODE_NAMES = {
    "html": "index.html", "css": "style.css", "js": "script.js",
    "ts": "script.ts", "py": "code.py", "json": "data.json",
    "sql": "query.sql", "sh": "script.sh", "md": "README.md",
    "Dockerfile": "Dockerfile", "txt": "code.txt",
}


def _safe_code_filename(name: str, fallback: str) -> str:
    """Не допускает путей и опасных/пустых имён из ответа модели."""
    name = os.path.basename((name or "").strip().strip('"\''))
    name = re.sub(r"[^0-9A-Za-zА-Яа-яЁё._+() -]", "_", name).strip(" .")
    if not name or name in {".", ".."}:
        return fallback
    return name[:100]


def _parse_code_info(info: str, index: int, total: int) -> tuple[str, str]:
    """Возвращает (язык, безопасное имя файла) из строки после ```."""
    parts = (info or "").strip().split()
    language = (parts[0].lower() if parts else "text").lstrip(".")
    extension = _CODE_EXTENSIONS.get(language, "txt")
    explicit = ""
    for token in parts[1:]:
        if token.lower().startswith(("filename=", "file=")):
            explicit = token.split("=", 1)[1]
            break
    default = _DEFAULT_CODE_NAMES.get(extension, f"code.{extension}")
    if total > 1 and not explicit:
        if default == "Dockerfile":
            default = f"Dockerfile_{index}"
        else:
            stem, dot, suffix = default.rpartition(".")
            default = f"{stem or 'code'}_{index}{dot}{suffix}" if dot else f"{default}_{index}"
    return language, _safe_code_filename(explicit, default)


def _unique_code_filename(filename: str, used: set[str]) -> str:
    candidate = filename
    n = 2
    while candidate.lower() in used:
        stem, ext = os.path.splitext(filename)
        candidate = f"{stem}_{n}{ext}"
        n += 1
    used.add(candidate.lower())
    return candidate


async def send_response(msg: Message, text: str, *, uid: Optional[int] = None) -> Optional[int]:
    """Отправляет пояснение текстом, а fenced-код — файлами, если настройка включена."""
    if uid is None or not user_code_files.get(uid, False) or _is_error_answer(text):
        return await _send_text_response(msg, text)

    matches = list(_CODE_FENCE_RE.finditer(text or ""))
    if not matches:
        return await _send_text_response(msg, text)

    prose = _CODE_FENCE_RE.sub("", text)
    prose = re.sub(r"\n{3,}", "\n\n", prose).strip()
    first_id: Optional[int] = None
    if prose:
        first_id = await _send_text_response(msg, prose)
    else:
        sent = await msg.answer("📎 Код из ответа отправлен отдельным файлом.")
        first_id = sent.message_id

    used_names: set[str] = set()
    for index, match in enumerate(matches, start=1):
        info, code = match.group(1), match.group(2).strip("\n")
        language, filename = _parse_code_info(info, index, len(matches))
        filename = _unique_code_filename(filename, used_names)
        try:
            document = BufferedInputFile(code.encode("utf-8"), filename=filename)
            sent = await msg.answer_document(
                document,
                caption=f"📎 Код: <code>{html.quote(filename)}</code>",
            )
            if first_id is None:
                first_id = sent.message_id
        except Exception as exc:
            logging.error("code file send error (%s): %s", filename, exc)
            # Если Telegram не принял документ, код не теряется — отправляем блоком.
            fallback = f"```{language}\n{code}\n```"
            fallback_id = await _send_text_response(msg, fallback)
            if first_id is None:
                first_id = fallback_id
    return first_id


def _split(text: str, mx: int) -> list[str]:
    if len(text) <= mx:
        return [text]
    parts = []
    while text:
        if len(text) <= mx:
            parts.append(text)
            break
        sp = mx
        for sep in ["\n\n", "\n", " "]:
            p = text.rfind(sep, 0, mx)
            if p > mx // 2:
                sp = p + len(sep)
                break
        parts.append(text[:sp])
        text = text[sp:]
    return parts


# ══════════════════════════════════════════════════════════════
# БАННЕРЫ МЕНЮ (картинки над экранами)
# ══════════════════════════════════════════════════════════════
# Экраны меню могут показываться как фото с подписью (баннер из
# MENU_IMAGE_DIR). Файла нет / подпись длиннее 1024 символов — тихо
# откатываемся на обычный текст, поведение как раньше.

# Кэш file_id: после первой загрузки картинка не гоняется по сети заново.
_menu_banner_file_id: dict[str, str] = {}


def _menu_image_path(name: str) -> str:
    """Путь к баннеру экрана (MENU_IMAGE_DIR/<name>.png) или '' если файла нет."""
    if not name:
        return ""
    p = os.path.join(MENU_IMAGE_DIR, f"{name}.png")
    return p if os.path.exists(p) else ""


def _banner_media(path: str):
    """file_id из кэша (после первой отправки) или локальный файл."""
    return _menu_banner_file_id.get(path) or FSInputFile(path)


def _banner_remember(path: str, m) -> None:
    try:
        if isinstance(m, Message) and m.photo:
            _menu_banner_file_id[path] = m.photo[-1].file_id
    except Exception:
        pass


async def _menu_send(msg: Message, text: str, markup, image: str = "") -> None:
    """Отправить экран меню новым сообщением: с баннером, если он есть
    и текст влезает в лимит подписи Telegram (1024 символа)."""
    path = _menu_image_path(image)
    if path and len(text) <= 1024:
        try:
            sent = await msg.answer_photo(_banner_media(path), caption=text,
                                          reply_markup=markup)
            _banner_remember(path, sent)
            return
        except Exception as e:
            logging.debug(f"menu banner send failed ({path}): {e}")
            _menu_banner_file_id.pop(path, None)
    await msg.answer(text, reply_markup=markup)


async def _menu_edit(msg: Message, text: str, markup, image: str = "") -> None:
    """Перерисовать экран меню на месте. Telegram не умеет превращать
    текстовое сообщение в фото и обратно, поэтому при смене типа (или
    неудачном редактировании) старое сообщение удаляется и шлётся новое."""
    path = _menu_image_path(image)
    want_photo = bool(path) and len(text) <= 1024
    is_photo = bool(msg.photo)
    try:
        if want_photo and is_photo:
            sent = await msg.edit_media(
                InputMediaPhoto(media=_banner_media(path), caption=text),
                reply_markup=markup,
            )
            _banner_remember(path, sent)
            return
        if not want_photo and not is_photo:
            await msg.edit_text(text, reply_markup=markup)
            return
    except TelegramBadRequest as e:
        if "not modified" in str(e).lower():
            return
        logging.debug(f"menu edit failed ({image}): {e}")
        if path:
            _menu_banner_file_id.pop(path, None)
    except Exception as e:
        logging.debug(f"menu edit failed ({image}): {e}")
    # Смена типа сообщения или редактирование не удалось — пересоздаём.
    try:
        await msg.delete()
    except Exception:
        pass
    await _menu_send(msg, text, markup, image=image)


# ══════════════════════════════════════════════════════════════
# КЛАВИАТУРЫ — ВЫБОР КАТЕГОРИИ МОДЕЛЕЙ
# ══════════════════════════════════════════════════════════════

def _category_kb(current: str, uid: int) -> InlineKeyboardBuilder:
    """Клавиатура выбора КАТЕГОРИИ — при нажатии показывает модели внутри."""
    b = InlineKeyboardBuilder()
    for key, cat in CATEGORIES.items():
        if category_wip_for(uid, key):
            label = f"🛠 {cat.emoji} {cat.name} (в разработке)"
        elif category_locked_for(uid, key):
            label = f"🔒 {cat.emoji} {cat.name}"
        else:
            check = "✅ " if key == current else ""
            label = f"{check}{cat.emoji} {cat.name}"
        b.row(InlineKeyboardButton(text=label, callback_data=f"category:show:{key}"))
    b.row(InlineKeyboardButton(text="⬅️ Назад", callback_data="model_back"))
    return b


def _category_info_text(current: str, uid: int) -> str:
    """Текст со списком категорий и объяснением каждой."""
    lines = ["🔀 <b>Категории моделей</b>\n━━━━━━━━━━━━━━━\n"]
    for key, cat in CATEGORIES.items():
        if category_wip_for(uid, key):
            lines.append(f"🛠 <b>{cat.name}</b> — в разработке\n{cat.desc}\n")
        elif category_locked_for(uid, key):
            lines.append(f"🔒 <b>{cat.name}</b> — только для Premium\n{cat.desc}\n")
        else:
            mark = " ✅" if key == current else ""
            lines.append(f"{cat.emoji} <b>{cat.name}</b>{mark}\n{cat.desc}\n")
    lines.append("<i>Нажмите на категорию, чтобы увидеть модели внутри.</i>")
    return "\n".join(lines)


def _model_list_kb(cat_key: str, uid: int, view_toggle: str = "") -> InlineKeyboardBuilder:
    """Клавиатура со списком моделей внутри категории.
    view_toggle: "full" — добавить кнопку «Подробнее» (раскрыть описания),
    "short" — кнопку «Кратко» (вернуться к баннеру), "" — без кнопки."""
    b = InlineKeyboardBuilder()
    specific = user_specific_model.get(uid)
    active_models = category_models(cat_key)
    auto_selected = specific is None or specific not in active_models
    auto_label = "🤖 Auto" + (" ✅" if auto_selected else "")
    b.row(InlineKeyboardButton(text=auto_label, callback_data=f"model:auto:{cat_key}"))
    overrides = _model_overrides()
    for mk in visible_category_models(cat_key):
        if mk in ADMIN_ONLY_MODEL_KEYS and not _is_admin(uid):
            continue
        m = MODELS[mk]
        check = " ✅" if specific == mk and mk in active_models else ""
        maintenance = overrides.get(mk, {}).get("state") == "maintenance"
        health_icon, _, _ = _model_health_view(mk)
        status_prefix = f"{health_icon} " if health_icon else ""
        crown = "👑 " if mk in ADMIN_ONLY_MODEL_KEYS else ""
        price = f" 💎×{m.token_coef:g}" if m.token_coef > 1 else ""
        label = (f"{status_prefix}{'🔧 ' if maintenance else ''}"
                 f"{crown}{m.emoji} {m.name}{price}{check}")
        b.row(InlineKeyboardButton(text=label, callback_data=f"model:select:{cat_key}:{mk}"))
    if view_toggle == "full":
        b.row(InlineKeyboardButton(text="ℹ️ Подробнее о моделях",
                                   callback_data=f"model:view:{cat_key}:full"))
    elif view_toggle == "short":
        b.row(InlineKeyboardButton(text="🖼 Кратко",
                                   callback_data=f"model:view:{cat_key}:short"))
    b.row(InlineKeyboardButton(text="⬅️ К категориям", callback_data="model_menu_back"))
    return b


def _model_list_text(cat_key: str, uid: int, compact: bool = False) -> str:
    """Текст со списком моделей внутри категории.
    compact=True — без описаний моделей (короткая подпись под баннером)."""
    cat = CATEGORIES[cat_key]
    specific = user_specific_model.get(uid)
    active_models = category_models(cat_key)
    lines = [
        f"{cat.emoji} <b>{cat.name}</b>\n━━━━━━━━━━━━━━━\n{cat.desc}\n",
        "Выберите модель или <b>Auto</b> — бот подберёт сам:\n",
    ]
    if specific and specific in active_models:
        lines.append(f"✅ Сейчас выбрана: <b>{MODELS[specific].name}</b>\n")
    elif not specific or specific not in category_models(cat_key):
        lines.append("✅ Сейчас: <b>Auto</b> (бот выбирает сам)\n")
    lines.append("")
    overrides = _model_overrides()
    for mk in visible_category_models(cat_key):
        if mk in ADMIN_ONLY_MODEL_KEYS and not _is_admin(uid):
            continue
        m = MODELS[mk]
        mark = " ✅" if specific == mk and mk in active_models else ""
        nv = " ⚠️NVIDIA" if m.nvidia else ""
        maintenance = overrides.get(mk, {}).get("state") == "maintenance"
        tech = " 🔧 На техработах" if maintenance else ""
        health_icon, error_percent, health_label = _model_health_view(mk)
        if health_icon and error_percent is not None:
            pretty_error = f"{error_percent:.1f}".rstrip("0").rstrip(".")
            health = f" {health_icon} {pretty_error}% ошибок"
        elif health_icon:
            health = f" {health_icon} {health_label}"
        else:
            health = ""
        desc = _model_description(mk)
        crown = "👑 " if mk in ADMIN_ONLY_MODEL_KEYS else ""
        price = f" · 💎 списание ×{m.token_coef:g}" if m.token_coef > 1 else ""
        lines.append(f"{crown}{m.emoji} <b>{m.name}</b>{mark}{nv}{tech}{health}{price}")
        if compact:
            continue
        if desc:
            lines.append(f"{html.quote(desc)}\n")
        else:
            lines.append("")
    if compact:
        lines.append("\n<i>Описания моделей — по кнопке «ℹ️ Подробнее»</i>")
    return "\n".join(lines)


async def _show_model_list(msg: Message, cat_key: str, uid: int, *,
                           full: bool = False, send: bool = False) -> None:
    """Экран моделей категории. Если есть баннер img/cat_<key>.png —
    показывает фото с компактной подписью (без описаний, если полный текст
    не влезает в 1024) и кнопкой «Подробнее»; иначе — обычный полный текст."""
    image = f"cat_{cat_key}"
    has_banner = bool(_menu_image_path(image))
    text = _model_list_text(cat_key, uid)
    toggle = ""
    if has_banner and not full and len(text) > 1024:
        compact = _model_list_text(cat_key, uid, compact=True)
        if len(compact) <= 1024:
            text, toggle = compact, "full"
    use_image = image if (has_banner and not full and len(text) <= 1024) else ""
    if has_banner and full:
        toggle = "short"
    markup = _model_list_kb(cat_key, uid, view_toggle=toggle).as_markup()
    if send:
        await _menu_send(msg, text, markup, image=use_image)
    else:
        await _menu_edit(msg, text, markup, image=use_image)


def _nvidia_confirm_kb(category_key: str) -> InlineKeyboardBuilder:
    b = InlineKeyboardBuilder()
    b.row(InlineKeyboardButton(text="✅ Я согласен", callback_data=f"nvidia_ok:{category_key}"))
    b.row(InlineKeyboardButton(text="⬅️ Назад", callback_data="model_menu_back"))
    return b


def _category_requires_nvidia_consent(cat_key: str, uid: int) -> bool:
    """True, если хотя бы одна модель категории требует NVIDIA-согласия,
    и пользователь его ещё не дал."""
    if nvidia_consent.get(uid, False):
        return False
    return any(MODELS[mk].nvidia for mk in category_models(cat_key))


# ══════════════════════════════════════════════════════════════
# ГЛАВНОЕ МЕНЮ (Reply Keyboard) И МАГАЗИН ТОКЕНОВ
# ════════════════════════════════════════════════════════════════

def main_menu_kb() -> InlineKeyboardMarkup:
    """Главное инлайн-меню с основными разделами бота (под приветствием)."""
    b = InlineKeyboardBuilder()
    b.row(InlineKeyboardButton(text=BTN_NEW_CHAT, callback_data="menu:new_chat"))
    b.row(
        InlineKeyboardButton(text=BTN_MODEL, callback_data="menu:model"),
        InlineKeyboardButton(text=BTN_SETTINGS, callback_data="menu:settings"),
    )
    b.row(
        InlineKeyboardButton(text=BTN_INVITE, callback_data="menu:invite"),
        InlineKeyboardButton(text=BTN_BONUS, callback_data="menu:bonus"),
    )
    b.row(
        InlineKeyboardButton(text=BTN_BUY, callback_data="menu:buy"),
        InlineKeyboardButton(text=BTN_STATS, callback_data="menu:stats"),
    )
    b.row(InlineKeyboardButton(text=BTN_HELP, callback_data="menu:help"))
    # Одна кнопка статуса вместо двух почти одинаковых («Проверить модели»
    # + «Статус моделей в боте» путали): ссылка на сайт есть внутри экрана
    # статуса (menu:health).
    b.row(
        InlineKeyboardButton(text="🌐 Сайт", url=SITE_URL),
        InlineKeyboardButton(text="🩺 Статус моделей", callback_data="menu:health"),
    )
    return b.as_markup()


def _menu_btn_kb() -> InlineKeyboardMarkup:
    """Компактная кнопка «Меню» под служебными сообщениями."""
    b = InlineKeyboardBuilder()
    b.row(InlineKeyboardButton(text="📋 Меню", callback_data="menu:main"))
    return b.as_markup()


def _shop_kb(uid: int) -> InlineKeyboardBuilder:
    """Inline-клавиатура магазина: Premium-подписка + пакеты токенов.

    Если подключена Platega — кнопка открывает выбор способа оплаты
    (Stars / рубли). Если Platega не настроена — прежнее поведение: сразу
    Счёт в Telegram Stars.
    """
    b = InlineKeyboardBuilder()
    pg = platega_enabled()
    # Пробный Premium: один раз, только пока Premium не активен.
    if TRIAL_DAYS > 0 and uid not in ADMIN_IDS and uid not in trial_used and not premium_active(uid):
        b.row(InlineKeyboardButton(
            text=f"🎁 Попробовать Premium бесплатно ({TRIAL_DAYS} дн.)",
            callback_data="trial:claim",
        ))
    # Premium доступен всем. Админам кнопку не показываем — у них Premium всегда.
    if uid not in ADMIN_IDS:
        pr = f"💎 Продлить Premium (+{PREMIUM_DAYS} дн.)" if premium_active(uid) \
            else f"💎 Premium на {PREMIUM_DAYS} дней"
        if pg:
            b.row(InlineKeyboardButton(
                text=f"{pr} — {_disc(PREMIUM_PRICE)}⭐ / {_disc(PREMIUM_PRICE_RUB)}₽",
                callback_data="pmenu:premium",
            ))
        else:
            b.row(InlineKeyboardButton(text=f"{pr} — {_disc(PREMIUM_PRICE)} ⭐", callback_data="buy_premium"))
    for pack in REQUEST_PACKS.values():
        note = f" · {pack.note}" if pack.note else ""
        if pg:
            b.row(InlineKeyboardButton(
                text=f"{pack.emoji} {fmt_tokens(pack.tokens)} токенов — {_disc(pack.stars)}⭐ / {_disc(pack.rub)}₽{note}",
                callback_data=f"pmenu:pack:{pack.key}",
            ))
        else:
            b.row(InlineKeyboardButton(
                text=f"{pack.emoji} {fmt_tokens(pack.tokens)} токенов — {_disc(pack.stars)} ⭐{note}",
                callback_data=f"shop:{pack.key}",
            ))
    b.row(InlineKeyboardButton(text="💫 Своя сумма ⭐", callback_data="topup:stars"))
    if pg:
        b.row(InlineKeyboardButton(text="💳 Своя сумма ₽", callback_data="topup:rub"))
    b.row(InlineKeyboardButton(text="⬅️ Назад", callback_data="back_start"))
    return b


def _shop_text(uid: int) -> str:
    left = remaining(uid)
    lines = [
        "💰 <b>Пополнение баланса</b>\n",
    ]
    banner = _sale_banner()
    if banner:
        lines.append(banner)
    lines.append(f"✨ Баланс: <b>{fmt_tokens(left)}</b> токенов")
    if premium_active(uid):
        lines.append(f"💎 Premium активен до <b>{premium_until[uid].strftime('%d.%m.%Y')}</b>")
    lines.append(
        f"\n📈 Курс: 1 ₽ = <b>{TOKENS_PER_RUB}</b> токенов · 1 ⭐ = <b>{TOKENS_PER_STAR}</b> токенов\n"
        f"Списание — за объём вопроса и ответа, минимум {MIN_TOKENS_SPEND} токенов за ответ.\n\n"
        "💎 <b>Premium</b>: категория «Быстрые», −20% на списание токенов, "
        "без рекламы и обязательной подписки.\n"
        "⭐ <b>Пакеты</b>: чем больше — тем выгоднее.\n"
    )
    lines.append("Выберите вариант:")
    return "\n".join(lines)


async def _show_shop(msg: Message, uid: int) -> None:
    await msg.answer(_shop_text(uid), reply_markup=_shop_kb(uid).as_markup())


# ══════════════════════════════════════════════════════════════
# ПРИВЕТСТВИЕ И ПОМОЩЬ
# ══════════════════════════════════════════════════════════════

def _cheapest_pack_rub() -> int:
    """Минимальная цена входа в рублях — для подзаголовка приветствия.

    Берётся из REQUEST_PACKS, а не хардкодом: поменяешь прайс — текст на первом
    экране обновится сам и не начнёт врать.
    """
    prices = [p.rub for p in REQUEST_PACKS.values() if getattr(p, "rub", 0)]
    return min(prices) if prices else 0


def _welcome_text(uid: int) -> str:
    left = remaining(uid)
    cat = CATEGORIES[user_category(uid)]
    return (
        "🤖 <b>AI-ассистент</b>\n"
        f"<i>{len(MODELS)} моделей от {_cheapest_pack_rub()} ₽ — дешевле подписки на ChatGPT</i>\n"
        "━━━━━━━━━━━━━━━\n"
        "💬 <b>Текст</b> — просто напишите вопрос\n"
        "📷 <b>Фото</b> — проанализирую изображение\n"
        "🎤 <b>Голос / кружок</b> — расшифрую и отвечу\n"
        "━━━━━━━━━━━━━━━\n"
        f"{cat.emoji} Категория: <b>{cat.name}</b>\n"
        f"💰 Баланс: <b>{fmt_tokens(left)}</b> токенов\n\n"
        "👉 <b>Просто напишите вопрос в чат — я отвечу.</b>\n"
        "<i>Остальное — на кнопках ниже.</i>"
    )


def _welcome_image() -> str:
    """Путь к приветственному баннеру (или '' если файла нет)."""
    if WELCOME_IMAGE and os.path.exists(WELCOME_IMAGE):
        return WELCOME_IMAGE
    if os.path.exists("welcome.png"):
        return "welcome.png"
    return ""


async def _send_welcome(msg: Message, uid: int) -> None:
    # У старых пользователей могла остаться нижняя Reply-клавиатура —
    # убираем её служебным сообщением, которое сразу удаляем.
    try:
        _tmp = await msg.answer("⌨️", reply_markup=ReplyKeyboardRemove())
        await _tmp.delete()
    except Exception:
        pass
    # Приветственный баннер (если файл есть). При любой ошибке — обычный текст,
    # чтобы приветствие никогда не «сломалось» из-за отсутствия/битой картинки.
    banner = _welcome_image()
    if banner:
        try:
            sent = await msg.answer_photo(
                _banner_media(banner),
                caption=_welcome_text(uid),
                reply_markup=main_menu_kb(),
            )
            _banner_remember(banner, sent)
            return
        except Exception as e:
            logging.debug(f"welcome banner send failed: {e}")
            _menu_banner_file_id.pop(banner, None)
    await msg.answer(_welcome_text(uid), reply_markup=main_menu_kb())


async def _edit_welcome(msg: Message, uid: int) -> None:
    """Перерисовывает приветствие на месте (без удаления сообщения)."""
    banner = _welcome_image()
    try:
        if banner and msg.photo:
            sent = await msg.edit_media(
                InputMediaPhoto(media=_banner_media(banner), caption=_welcome_text(uid)),
                reply_markup=main_menu_kb(),
            )
            _banner_remember(banner, sent)
            return
        if not banner and not msg.photo:
            await msg.edit_text(_welcome_text(uid), reply_markup=main_menu_kb())
            return
    except TelegramBadRequest as e:
        if "not modified" in str(e).lower():
            return
        logging.debug(f"welcome edit failed: {e}")
        if banner:
            _menu_banner_file_id.pop(banner, None)
    except Exception as e:
        logging.debug(f"welcome edit failed: {e}")
    # Смена типа сообщения или редактирование не удалось — пересоздаём.
    try:
        await msg.delete()
    except Exception:
        pass
    await _send_welcome(msg, uid)


async def _show_help(msg: Message, uid: int) -> None:
    left = remaining(uid)
    await msg.answer(
        "❓ <b>Помощь</b>\n\n"
        "Я — AI-ассистент. Вот что я умею:\n\n"
        "💬 <b>Текст</b> — напишите вопрос, и я отвечу\n"
        "📷 <b>Фото</b> — пришлите изображение, я его проанализирую\n"
        "🎤 <b>Голос / видеокружок</b> — расшифрую и отвечу\n\n"
        "<b>Разделы меню:</b>\n"
        f"{BTN_NEW_CHAT} — очистить историю и начать заново\n"
        f"{BTN_MODEL} — выбрать категорию моделей\n"
        f"{BTN_SETTINGS} — настройки\n"
        f"{BTN_BUY} — пополнить баланс токенов\n"
        f"{BTN_STATS} — ваш баланс и статистика\n"
        f"{BTN_INVITE} — бонус за приглашённых друзей\n"
        f"{BTN_BONUS} — бесплатные токены каждый день\n\n"
        f"💰 Баланс: <b>{fmt_tokens(left)}</b> токенов\n\n"
        f"🆘 <b>Поддержка:</b> {SUPPORT_USERNAME}",
        reply_markup=_menu_btn_kb(),
    )


# ══════════════════════════════════════════════════════════════
# КОМАНДЫ
# ════════════════════════════════════════════════════════════════

def _consent_intro_kb() -> InlineKeyboardBuilder:
    """Кнопки экрана согласия: ссылки на документы + «Принимаю»."""
    b = InlineKeyboardBuilder()
    b.row(InlineKeyboardButton(text="📄 Пользовательское соглашение", url=USER_AGREEMENT_URL))
    b.row(InlineKeyboardButton(text="🔒 Политика конфиденциальности", url=PRIVACY_POLICY_URL))
    b.row(InlineKeyboardButton(text="🆘 Поддержка", url=SUPPORT_URL))
    b.row(InlineKeyboardButton(text="✅ Принимаю", callback_data="consent:accept", style="success"))
    return b


# _consent_confirm_kb и _captcha_kb удалены 22.08.2026 вместе с экраном
# «вы точно прочитали?» и робо-проверкой: они добавляли 2-3 нажатия между
# /start и первым ответом, ничего при этом не защищая (капча с 3 вариантами
# угадывается с вероятностью 1/3, а согласие юридически получено нажатием
# «Принимаю»). Множество captcha_solved осталось в состоянии и в базе —
# оно больше не читается, но и не мешает загрузке старых данных.


_CONSENT_INTRO_TEXT = (
    "👋 <b>Добро пожаловать!</b>\n\n"
    "Перед началом использования, пожалуйста, ознакомьтесь с документами "
    "(кнопки ниже):\n\n"
    "📄 <b>Пользовательское соглашение</b>\n"
    "🔒 <b>Политика конфиденциальности</b>\n\n"
    "Нажимая «Принимаю», вы подтверждаете, что прочитали их и согласны.\n\n"
    "ℹ️ Эти документы всегда доступны в разделе ⚙️ <b>Настройки</b>."
)


async def _send_consent_intro(msg: Message) -> None:
    await msg.answer(_CONSENT_INTRO_TEXT, reply_markup=_consent_intro_kb().as_markup())


def _parse_ref_payload(text: str) -> Optional[int]:
    """Достаёт uid пригласившего из deep-link '/start ref_<uid>'."""
    parts = (text or "").split(maxsplit=1)
    if len(parts) < 2:
        return None
    payload = parts[1].strip()
    if payload.startswith("ref_") and payload[4:].isdigit():
        return int(payload[4:])
    return None


@router.message(CommandStart())
async def cmd_start(msg: Message) -> None:
    uid = _uid(msg)

    # Deep-link реферала: запоминаем пригласившего ТОЛЬКО для нового пользователя
    # (ещё не принявшего соглашение). Награда выдаётся после согласия — см.
    # cb_consent_accept → _award_referral_if_any. Существующим юзерам ссылка ничего
    # не даёт (иначе можно было бы фармить награды перезаходами).
    is_new = uid not in ADMIN_IDS and not agreement_accepted.get(uid)
    if is_new:
        ref_uid = _parse_ref_payload(msg.text)
        if ref_uid is not None and _valid_referrer(uid, ref_uid):
            pending_referral[uid] = ref_uid

    # Новому пользователю показываем ТОЛЬКО экран согласия — оно нужно платёжной
    # системе. Ни рекламы, ни капчи на входе: первое сообщение в жизни юзера не
    # должно быть постом спонсора, иначе он уходит, не увидев бота.
    # ОП здесь тоже НЕ показываем: первый ответ у новичка бесплатный, а ОП
    # включается на самом ответе (см. _run_ai_turn). Админов не беспокоим.
    if is_new:
        await _send_consent_intro(msg)
        return

    # Приветственный пост BotoHub — только для уже согласившихся, не на онбординге.
    await _botohub_hi(uid)
    await _send_welcome(msg, uid)


# Согласие принимается ОДНИМ нажатием «Принимаю». Ни капчи, ни повторного
# вопроса «вы точно прочитали?»: юридически согласие получено самим нажатием,
# а каждый лишний экран между /start и первым ответом стоит части пользователей.
# consent:yes оставлен как алиас — на кнопки в старых сообщениях у тех, кто
# застрял на прежнем онбординге.
@router.callback_query(F.data.in_({"consent:accept", "consent:yes"}))
async def cb_consent_accept(cb: CallbackQuery) -> None:
    uid = cb.from_user.id
    # Гасим «часики» первым делом: всё остальное здесь — сетевые вызовы, и любой
    # из них может упасть (двойной тап → «message is not modified», старое или
    # удалённое сообщение). Раньше cb.answer() стоял последним, и тогда падение
    # edit_text оставляло человека на экране согласия без меню: в базе согласие
    # уже есть, приветствие не пришло, повторные нажатия дают ту же ошибку.
    await cb.answer()
    agreement_accepted[uid] = True
    # Согласие фиксируем на диск сразу: это юридически значимое действие и
    # условие работы платёжной системы, его нельзя потерять при падении в
    # окне автосейва.
    save_state_now()
    # Реферальные награды (если пришёл по ссылке) — после согласия, чтобы
    # исключить накрутку незавершёнными регистрациями.
    await _award_referral_if_any(cb.bot, uid)
    got_ref_bonus = uid in invited_by
    thanks = "✅ Спасибо! Приятного пользования 🚀"
    if got_ref_bonus:
        thanks += f"\n\n🎁 Вам начислено <b>{fmt_tokens(REF_WELCOME)}</b> токенов за переход по приглашению!"
    try:
        await cb.message.edit_text(thanks)
    except Exception as e:
        # Не смогли переписать экран согласия — не страшно, главное дальше
        # отдать приветствие, иначе человек останется без меню.
        logging.debug(f"consent: не удалось обновить экран (uid={uid}): {e}")
    await _send_welcome(cb.message, uid)


async def _consent_guard(handler, event, data):
    """Пока новый пользователь не принял соглашение — не пускаем к боту.

    Работает на все сообщения. Исключения: админы, команда /start
    (её обрабатывает cmd_start, показывая экран согласия) и callback-кнопки
    consent:* (middleware ставится только на message, callback'и проходят).
    """
    user = getattr(event, "from_user", None)
    uid = user.id if user else None
    # successful_payment — это Message, но деньги по нему УЖЕ списаны Telegram.
    # Блокировать его нельзя ни при каком гварде: иначе «деньги взяты, товар
    # не выдан» без самовосстановления. Пропускаем всегда (возврат — руками
    # через админку, кнопка возврата ⭐).
    if getattr(event, "successful_payment", None) is not None:
        return await handler(event, data)
    if uid is not None and project_closed and uid not in ADMIN_IDS:
        # Проект закрыт: не-админам один раз на /start показываем сообщение
        # и закрепляем его, дальше игнорируем молча (бот работает только у админов).
        text = getattr(event, "text", "") or ""
        # Маркер ставим ТОЛЬКО когда сообщение действительно показали (22.08.2026).
        # Раньше add(uid) стоял до проверки на /start: человек писал «привет»,
        # помечался как «уже уведомлён», а его следующий /start уходил в молчание
        # навсегда — он так и не узнавал, что проект закрыт.
        if text.startswith("/start") and uid not in project_closed_notified:
            project_closed_notified.add(uid)
            try:
                msg = await event.answer(
                    "<b>🔴 Проект закрыт.</b>\n\n"
                    "К сожалению, работа бота приостановлена. "
                    "Следите за обновлениями — возможно, проект ещё вернётся."
                )
                await msg.pin(disable_notification=True)
            except Exception as e:
                logging.debug(f"closed: сообщение не доставлено (uid={uid}): {e}")
        return  # остальные сообщения закрытого проекта игнорируются молча
    if uid is not None and uid in banned_users and uid not in ADMIN_IDS:
        # Забаненным на /start один раз показываем сообщение о блокировке
        # и закрепляем его в чате (видно обеим сторонам).
        text = getattr(event, "text", "") or ""
        # Тот же порядок, что и у закрытого проекта: маркер только после показа,
        # иначе забаненный, написавший что-то кроме /start, никогда не узнает
        # причину — для него бот просто «сломался».
        if text.startswith("/start") and uid not in banned_notified:
            banned_notified.add(uid)
            try:
                msg = await event.answer(
                    "<b>🚫 Вы были заблокированы.</b>\n\n"
                    "Доступ к боту закрыт. Если считаете, что это ошибка — "
                    "свяжитесь с администратором."
                )
                await msg.pin(disable_notification=True)
            except Exception as e:
                logging.debug(f"ban: сообщение не доставлено (uid={uid}): {e}")
        return  # забаненные игнорируются молча
    if uid is not None and uid not in ADMIN_IDS and not agreement_accepted.get(uid):
        text = getattr(event, "text", "") or ""
        if text.startswith("/start"):
            return await handler(event, data)  # cmd_start сам покажет согласие
        # Гвард отвечает только за согласие; ОП теперь на самом ответе (_run_ai_turn).
        await _send_consent_intro(event)
        return  # блокируем дальнейшую обработку до принятия соглашения
    return await handler(event, data)


router.message.outer_middleware(_consent_guard)
# edited_message — ОТДЕЛЬНЫЙ обсервер aiogram, middleware выше на него НЕ
# действует. Без этой строки забаненный / не принявший соглашение / юзер
# закрытого проекта редактировал своё старое сообщение и получал ответ
# модели в обход всех гвардов.
router.edited_message.outer_middleware(_consent_guard)


async def _ban_guard_cb(handler, event, data):
    """Блокирует callback-кнопки для забаненных пользователей."""
    user = getattr(event, "from_user", None)
    if user and user.id in banned_users and user.id not in ADMIN_IDS:
        try:
            await event.answer("⛔ Доступ ограничен.", show_alert=True)
        except Exception:
            pass
        return
    if user and project_closed and user.id not in ADMIN_IDS:
        try:
            await event.answer("🔴 Проект закрыт.", show_alert=True)
        except Exception:
            pass
        return
    return await handler(event, data)


router.callback_query.outer_middleware(_ban_guard_cb)


@router.edited_message()
async def on_edited(msg: Message, bot: Bot) -> None:
    if not msg.text:
        return
    uid = _uid(msg)
    if admin_state.get(uid) == "waiting_broadcast":
        return
    # _track_user раньше _check_limit: именно он выдаёт новичку стартовые токены.
    # В обратном порядке человек, который отредактировал своё самое первое
    # сообщение, получал «токены закончились» при нулевом балансе. В основном
    # текстовом хендлере порядок именно такой.
    _track_user(uid, msg.from_user)
    if not await _check_limit(msg, uid):
        return
    histories.pop(uid, None)

    chat_id = msg.chat.id
    old_bot_id = user_msg_to_bot_msg.get(chat_id, {}).get(msg.message_id)

    await _run_ai_turn(msg, bot, msg.text)

    if old_bot_id is not None:
        try:
            await bot.delete_message(chat_id=chat_id, message_id=old_bot_id)
        except Exception:
            pass


# ── Общие действия разделов (используются и командами, и кнопками) ──

async def _new_chat(msg: Message, uid: int) -> None:
    histories.pop(uid, None)
    await msg.answer(
        "🤖 <b>Новый чат начат!</b>\n"
        "История очищена — спрашивайте что угодно.",
        reply_markup=_menu_btn_kb(),
    )


async def _show_model_menu(msg: Message, uid: int) -> None:
    model_menu_source[uid] = "main"   # открыто из главного меню/команды
    current = user_category(uid)
    await _refresh_model_health()
    await _menu_send(msg, _category_info_text(current, uid),
                     _category_kb(current, uid).as_markup(), image="cat_menu")


async def _show_limit(msg: Message, uid: int) -> None:
    u = _get_usage(uid)
    left = remaining(uid)
    plan = user_plan(uid)
    premium_line = (
        f"💎 Premium до: <b>{premium_until[uid].strftime('%d.%m.%Y')}</b>\n"
        if premium_active(uid) else ""
    )
    bonus_line = (
        "🎯 Ежедневный бонус: <b>доступен!</b> (главное меню → 🎯 Ежедневный бонус)\n"
        if can_claim_daily_bonus(uid)
        else "🎯 Ежедневный бонус: получен, приходите завтра\n"
    )
    ref_line = (
        f"👥 Приглашено друзей: <b>{referral_count[uid]}</b>\n"
        if referral_count.get(uid) else ""
    )
    await msg.answer(
        "📊 <b>Ваша статистика</b>\n\n"
        f"💳 Тариф: <b>{plan.name}</b>\n"
        f"{premium_line}"
        f"💰 Баланс: <b>{fmt_tokens(left)}</b> токенов\n"
        f"🔥 Потрачено сегодня: <b>{fmt_tokens(u['used'])}</b> токенов\n\n"
        f"{bonus_line}"
        f"{ref_line}",
        reply_markup=_menu_btn_kb(),
    )


@router.message(Command("clear"))
async def cmd_clear(msg: Message) -> None:
    await _new_chat(msg, _uid(msg))


@router.message(Command("limit"))
async def cmd_limit(msg: Message) -> None:
    await _show_limit(msg, _uid(msg))


@router.message(Command("model"))
async def cmd_model(msg: Message) -> None:
    await _show_model_menu(msg, _uid(msg))


@router.message(Command("settings"))
async def cmd_settings(msg: Message) -> None:
    await _send_settings(msg, _uid(msg))


@router.message(Command("hide"))
async def cmd_hide(msg: Message) -> None:
    """Скрывает нижнюю клавиатуру с кнопками."""
    await msg.answer(
        "⌨️ Кнопки скрыты. Чтобы вернуть их — команда /menu.",
        reply_markup=ReplyKeyboardRemove(),
    )


@router.message(Command("menu"))
async def cmd_menu(msg: Message) -> None:
    """Открывает главное инлайн-меню."""
    await msg.answer("📋 <b>Меню</b> — выберите раздел 👇", reply_markup=main_menu_kb())


def _settings_text(uid: int) -> str:
    cat = CATEGORIES[user_category(uid)]
    persona = user_persona_obj(uid)
    verb = user_verbosity_key(uid)
    code_files_state = "включено" if user_code_files.get(uid, False) else "выключено"
    return (
        "⚙️ <b>Настройки</b>\n\n"
        f"{cat.emoji} Категория: <b>{cat.name}</b>\n"
        f"{persona.emoji} Персона: <b>{persona.name}</b>\n"
        f"📏 Длина ответа: <b>{VERBOSITY_LABELS[verb]}</b>\n"
        f"📎 Код отдельными файлами: <b>{code_files_state}</b>\n\n"
        f"🆘 Поддержка: {SUPPORT_USERNAME}"
    )


def _settings_kb(uid: int) -> InlineKeyboardBuilder:
    persona = user_persona_obj(uid)
    verb = user_verbosity_key(uid)
    prompt_state = "задан" if user_custom_prompt.get(uid) else "не задан"
    code_files_state = "включено" if user_code_files.get(uid, False) else "выключено"
    b = InlineKeyboardBuilder()
    b.row(
        InlineKeyboardButton(text="🔀 Категория", callback_data="model_menu"),
        InlineKeyboardButton(text="🎟 Промокод", callback_data="settings:promo"),
    )
    b.row(InlineKeyboardButton(text=f"🎭 Персона: {persona.name}", callback_data="settings:persona"))
    b.row(InlineKeyboardButton(text=f"📏 Длина ответа: {VERBOSITY_LABELS[verb]}", callback_data="settings:verbosity"))
    b.row(InlineKeyboardButton(text=f"📝 Свой промпт (Premium): {prompt_state}", callback_data="settings:prompt"))
    b.row(InlineKeyboardButton(
        text=f"📎 Код отдельными файлами: {code_files_state}",
        callback_data="settings:code_files",
    ))
    b.row(
        InlineKeyboardButton(text="✅ Соглашение", url=USER_AGREEMENT_URL),
        InlineKeyboardButton(text="🔒 Приватность", url=PRIVACY_POLICY_URL),
    )
    b.row(
        InlineKeyboardButton(text="🆘 Поддержка", url=SUPPORT_URL),
        InlineKeyboardButton(text="⬅️ Назад", callback_data="back_start"),
    )
    return b


async def _send_settings(msg: Message, uid: int) -> None:
    await msg.answer(_settings_text(uid), reply_markup=_settings_kb(uid).as_markup())


async def _edit_settings(msg: Message, uid: int) -> None:
    """Перерисовывает настройки на месте (без удаления сообщения)."""
    try:
        await msg.edit_text(_settings_text(uid), reply_markup=_settings_kb(uid).as_markup())
        return
    except TelegramBadRequest as e:
        if "not modified" in str(e).lower():
            return
        logging.debug(f"settings edit failed: {e}")
    except Exception as e:
        logging.debug(f"settings edit failed: {e}")
    # Не удалось отредактировать (например, сообщение-фото) — пересоздаём.
    try:
        await msg.delete()
    except Exception:
        pass
    await _send_settings(msg, uid)


@router.callback_query(F.data == "settings:code_files")
async def cb_settings_code_files(cb: CallbackQuery) -> None:
    uid = cb.from_user.id
    enabled = not user_code_files.get(uid, False)
    user_code_files[uid] = enabled
    save_state()
    await cb.answer(
        "📎 Код будет отправляться файлами" if enabled
        else "📎 Код снова будет отображаться в сообщении"
    )
    await _edit_settings(cb.message, uid)


@router.callback_query(F.data == "settings:promo")
async def cb_settings_promo(cb: CallbackQuery) -> None:
    """Активация промокода из настроек: просим прислать код сообщением."""
    user_input_state[cb.from_user.id] = "promo_code"
    await cb.answer()
    await cb.message.answer(
        "🎟 <b>Активация промокода</b>\n\n"
        "Отправьте промокод одним сообщением.\n\n"
        "Отмена — /cancel",
    )


@router.callback_query(F.data == "settings:menu")
async def cb_settings_menu(cb: CallbackQuery) -> None:
    """Возврат в меню настроек: перерисовываем настройки на месте."""
    await cb.answer()
    await _edit_settings(cb.message, cb.from_user.id)


# ── Персона ────────────────────────────────────────────────────

def _persona_kb(uid: int) -> InlineKeyboardBuilder:
    current = user_persona_obj(uid).key
    b = InlineKeyboardBuilder()
    for key, p in PERSONAS.items():
        mark = " ✅" if key == current else ""
        b.row(InlineKeyboardButton(text=f"{p.emoji} {p.name}{mark}", callback_data=f"persona:set:{key}"))
    b.row(InlineKeyboardButton(text="⬅️ К настройкам", callback_data="settings:menu"))
    return b


@router.callback_query(F.data == "settings:persona")
async def cb_settings_persona(cb: CallbackQuery) -> None:
    await cb.answer()
    try:
        await cb.message.edit_text(
            "🎭 <b>Персона ассистента</b>\n\n"
            "Выберите характер, в котором бот будет отвечать:",
            reply_markup=_persona_kb(cb.from_user.id).as_markup(),
        )
    except Exception:
        pass


@router.callback_query(F.data.startswith("persona:set:"))
async def cb_persona_set(cb: CallbackQuery) -> None:
    uid = cb.from_user.id
    key = cb.data.split(":", 2)[2]
    if key not in PERSONAS:
        await cb.answer("Неизвестная персона")
        return
    user_persona[uid] = key
    save_state()
    await cb.answer(f"{PERSONAS[key].emoji} Персона: {PERSONAS[key].name}", show_alert=False)
    try:
        await cb.message.edit_text(
            "🎭 <b>Персона ассистента</b>\n\n"
            "Выберите характер, в котором бот будет отвечать:",
            reply_markup=_persona_kb(uid).as_markup(),
        )
    except Exception:
        pass


# ── Длина ответа ───────────────────────────────────────────────

def _verbosity_kb(uid: int) -> InlineKeyboardBuilder:
    current = user_verbosity_key(uid)
    b = InlineKeyboardBuilder()
    for key in ("short", "medium", "long"):
        mark = " ✅" if key == current else ""
        b.row(InlineKeyboardButton(text=f"{VERBOSITY_LABELS[key]}{mark}", callback_data=f"verb:set:{key}"))
    b.row(InlineKeyboardButton(text="⬅️ К настройкам", callback_data="settings:menu"))
    return b


@router.callback_query(F.data == "settings:verbosity")
async def cb_settings_verbosity(cb: CallbackQuery) -> None:
    await cb.answer()
    try:
        await cb.message.edit_text(
            "📏 <b>Длина ответа</b>\n\n"
            "✂️ <b>Кратко</b> — 1–3 предложения, только суть\n"
            "📄 <b>Средне</b> — обычный ответ\n"
            "📚 <b>Подробно</b> — развёрнуто, с примерами",
            reply_markup=_verbosity_kb(cb.from_user.id).as_markup(),
        )
    except Exception:
        pass


@router.callback_query(F.data.startswith("verb:set:"))
async def cb_verbosity_set(cb: CallbackQuery) -> None:
    uid = cb.from_user.id
    key = cb.data.split(":", 2)[2]
    if key not in VERBOSITY_TOKENS:
        await cb.answer("Неизвестный режим")
        return
    user_verbosity[uid] = key
    save_state()
    await cb.answer(f"Длина: {VERBOSITY_LABELS[key]}")
    try:
        await cb.message.edit_text(
            "📏 <b>Длина ответа</b>\n\n"
            "✂️ <b>Кратко</b> — 1–3 предложения, только суть\n"
            "📄 <b>Средне</b> — обычный ответ\n"
            "📚 <b>Подробно</b> — развёрнуто, с примерами",
            reply_markup=_verbosity_kb(uid).as_markup(),
        )
    except Exception:
        pass


# ── Свой системный промпт (Premium) ────────────────────────

def _prompt_kb(uid: int) -> InlineKeyboardBuilder:
    b = InlineKeyboardBuilder()
    b.row(InlineKeyboardButton(text="✏️ Задать / изменить", callback_data="prompt:edit"))
    if user_custom_prompt.get(uid):
        b.row(InlineKeyboardButton(text="🗑 Сбросить", callback_data="prompt:clear"))
    b.row(InlineKeyboardButton(text="⬅️ К настройкам", callback_data="settings:menu"))
    return b


@router.callback_query(F.data == "settings:prompt")
async def cb_settings_prompt(cb: CallbackQuery) -> None:
    uid = cb.from_user.id
    if not is_premium(uid):
        await cb.answer("📝 Свой промпт доступен на Premium 💎", show_alert=True)
        return
    await cb.answer()
    current = user_custom_prompt.get(uid)
    cur_line = f"Текущий промпт:\n<i>{html.quote(current)}</i>" if current else "Промпт пока не задан."
    try:
        await cb.message.edit_text(
            "📝 <b>Свой промпт</b>\n\n"
            "Личная инструкция, которую бот будет учитывать в каждом ответе "
            "(стиль, формат, роль и т.п.).\n\n" + cur_line,
            reply_markup=_prompt_kb(uid).as_markup(),
        )
    except Exception:
        pass


@router.callback_query(F.data == "prompt:edit")
async def cb_prompt_edit(cb: CallbackQuery) -> None:
    uid = cb.from_user.id
    if not is_premium(uid):
        await cb.answer("Доступно на Premium 💎", show_alert=True)
        return
    user_input_state[uid] = "custom_prompt"
    await cb.answer()
    try:
        await cb.message.edit_text(
            "✏️ Пришлите текст промпта одним сообщением "
            f"(до {CUSTOM_PROMPT_MAX_LEN} символов).\n\n"
            "Отмена — /cancel",
        )
    except Exception:
        pass


@router.callback_query(F.data == "prompt:clear")
async def cb_prompt_clear(cb: CallbackQuery) -> None:
    uid = cb.from_user.id
    user_custom_prompt.pop(uid, None)
    save_state()
    await cb.answer("Промпт сброшен ✅", show_alert=True)
    await _edit_settings(cb.message, uid)


@router.message(Command("cancel"))
async def cmd_cancel(msg: Message) -> None:
    uid = _uid(msg)
    had = user_input_state.pop(uid, None) or admin_state.pop(uid, None)
    await msg.answer("✅ Отменено." if had else "Нечего отменять 🙂", reply_markup=_menu_btn_kb())


# ── Пригласить друга (реферальная ссылка) ──────────────────────

def _milestones_text(count: int) -> str:
    """Список реферальных уровней с отметками достигнутых."""
    lines = []
    for level in sorted(REF_MILESTONES):
        mark = "✅" if count >= level else "▫️"
        lines.append(f"{mark} {level} друзей — +{fmt_tokens(REF_MILESTONES[level])} токенов")
    return "\n".join(lines)


async def _send_invite(msg: Message, uid: int, bot: Bot) -> None:
    me = await bot.get_me()
    link = f"https://t.me/{me.username}?start=ref_{uid}"
    count = referral_count.get(uid, 0)
    earned = count * REF_REWARD
    from urllib.parse import quote as _q
    share_note = (
        "Нейросети прямо в Telegram — без VPN и зарубежных карт. "
        f"Переходи по моей ссылке — получишь {fmt_tokens(REF_WELCOME)} токенов в подарок!"
    )
    kb = InlineKeyboardBuilder()
    kb.row(InlineKeyboardButton(
        text="📤 Поделиться ссылкой",
        url="https://t.me/share/url?url=" + _q(link, safe="") + "&text=" + _q(share_note, safe=""),
    ))
    try:  # copy_text: Bot API 7.11+ / aiogram 3.15+; на старых версиях кнопки просто не будет
        from aiogram.types import CopyTextButton
        kb.row(InlineKeyboardButton(text="📋 Скопировать ссылку", copy_text=CopyTextButton(text=link)))
    except ImportError:
        pass
    kb.row(InlineKeyboardButton(text="📋 Меню", callback_data="menu:main"))
    await msg.answer(
        "🎁 <b>Пригласите друга — получите токены!</b>\n\n"
        f"За каждого друга, который перейдёт по вашей ссылке и начнёт "
        f"пользоваться ботом, вы получаете <b>+{fmt_tokens(REF_REWARD)}</b> токенов, "
        f"а друг — <b>+{REF_WELCOME}</b> в подарок.\n\n"
        f"🔗 <b>Ваша ссылка:</b>\n<code>{link}</code>\n\n"
        f"👥 Приглашено друзей: <b>{count}</b>\n"
        f"⭐ Заработано токенов: <b>{fmt_tokens(earned)}</b>\n\n"
        f"🏆 <b>Бонусы за уровни:</b>\n{_milestones_text(count)}",
        reply_markup=kb.as_markup(),
    )


# ── Ежедневный бонус ───────────────────────────────────────────

async def _claim_daily(msg: Message, uid: int) -> None:
    """Выдаёт ежедневный бонус (раз в сутки) с ростом по стрику.

    Антиабуз: бонус доступен только после реального запроса за день (см. _bonus_requires_activity)."""
    if not can_claim_daily_bonus(uid):
        await msg.answer(
            "🙌 <b>Сегодня бонус уже получен!</b>\n\n"
            "Возвращайтесь завтра — стрик продолжится, а бонус станет больше 🔥",
            reply_markup=_menu_btn_kb(),
        )
        return

    if _bonus_requires_activity(uid):
        await msg.answer(
            "🎯 <b>Бонус почти ваш!</b>\n\n"
            "Чтобы получить ежедневный бонус, задайте боту хотя бы один вопрос "
            "сегодня — и возвращайтесь за токенами. Так мы отсекаем ботов-фермеров 🤖",
            reply_markup=_menu_btn_kb(),
        )
        return

    # Стрик: +1, если бонус брали вчера, иначе сброс на 1.
    yesterday = (date.today() - timedelta(days=1)).isoformat()
    streak = daily_bonus_streak.get(uid, 0) + 1 if daily_bonus_date.get(uid) == yesterday else 1
    daily_bonus_streak[uid] = streak
    amount = _daily_bonus_amount(streak)
    _grant_requests(uid, amount)
    daily_bonus_date[uid] = _today_iso()
    save_state()

    at_cap = streak >= len(DAILY_BONUS_STREAK)
    streak_line = (
        f"🔥 Стрик: <b>{streak} дн.</b> — максимум, дальше бонус фиксированный\n"
        if at_cap
        else f"🔥 Стрик: <b>{streak} дн.</b> подряд — завтра будет ещё больше!\n"
    )
    await msg.answer(
        f"🎉 <b>Ежедневный бонус получен!</b>\n\n"
        f"➕ Начислено: <b>{fmt_tokens(amount)}</b> токенов\n"
        f"{streak_line}"
        f"📊 Доступно сейчас: <b>{fmt_count(remaining(uid))}</b>",
        reply_markup=_menu_btn_kb(),
    )


@router.callback_query(F.data == "model_menu")
async def cb_model_menu(cb: CallbackQuery) -> None:
    uid = cb.from_user.id
    model_menu_source[uid] = "settings"   # открыто из настроек
    current = user_category(uid)
    await cb.answer()
    await _refresh_model_health()
    await _menu_edit(cb.message, _category_info_text(current, uid),
                     _category_kb(current, uid).as_markup(), image="cat_menu")


@router.callback_query(F.data == "model_menu_back")
async def cb_model_menu_back(cb: CallbackQuery) -> None:
    """Возврат к списку категорий из списка моделей/NVIDIA-подтверждения.
    Источник открытия меню (main/settings) не трогаем — «Назад» из категорий
    вернёт туда, откуда пользователь пришёл."""
    uid = cb.from_user.id
    current = user_category(uid)
    await cb.answer()
    await _refresh_model_health()
    await _menu_edit(cb.message, _category_info_text(current, uid),
                     _category_kb(current, uid).as_markup(), image="cat_menu")


@router.callback_query(F.data == "model_back")
async def cb_model_back(cb: CallbackQuery) -> None:
    """«Назад» из списка категорий: возвращаем туда, откуда открыли меню —
    в настройки или в главное меню (перерисовываем на месте)."""
    uid = cb.from_user.id
    await cb.answer()
    if model_menu_source.get(uid) == "settings":
        await _edit_settings(cb.message, uid)
    else:
        await _edit_welcome(cb.message, uid)


@router.callback_query(F.data.startswith("category:show:"))
async def cb_category_show(cb: CallbackQuery) -> None:
    uid = cb.from_user.id
    key = cb.data.split(":", 2)[2]
    if key not in CATEGORIES:
        await cb.answer("Неизвестная категория")
        return

    if category_wip_for(uid, key):
        await cb.answer(WIP_NOTICE, show_alert=True)
        return

    if category_locked_for(uid, key):
        await cb.answer(
            f"🔒 «{CATEGORIES[key].name}» доступна только для Premium.",
            show_alert=True,
        )
        return

    if _category_requires_nvidia_consent(key, uid):
        await cb.answer()
        await _menu_edit(cb.message, NVIDIA_WARNING, _nvidia_confirm_kb(key).as_markup())
        return

    user_categories[uid] = key
    await cb.answer()
    await _refresh_model_health()
    await _show_model_list(cb.message, key, uid)


@router.callback_query(F.data.startswith("category:"))
async def cb_category_legacy(cb: CallbackQuery) -> None:
    """Старый callback (без :show:) — на случай если где-то остался."""
    uid = cb.from_user.id
    key = cb.data.split(":", 1)[1]
    if key not in CATEGORIES:
        await cb.answer("Неизвестная категория")
        return
    if category_wip_for(uid, key):
        await cb.answer(WIP_NOTICE, show_alert=True)
        return
    # Раньше легаси-путь пропускал и Premium-лок, и NVIDIA-согласие —
    # через старые кнопки можно было уйти в залоченную категорию.
    if category_locked_for(uid, key):
        await cb.answer(
            f"🔒 «{CATEGORIES[key].name}» доступна только для Premium.",
            show_alert=True,
        )
        return
    if _category_requires_nvidia_consent(key, uid):
        await cb.answer()
        await _menu_edit(cb.message, NVIDIA_WARNING, _nvidia_confirm_kb(key).as_markup())
        return
    user_categories[uid] = key
    cat = CATEGORIES[key]
    await cb.answer(f"✅ Категория: {cat.name}")  # тост вместо модального окна
    await _show_model_list(cb.message, key, uid)


@router.callback_query(F.data.startswith("model:"))
async def cb_model_select(cb: CallbackQuery) -> None:
    uid = cb.from_user.id
    parts = cb.data.split(":")
    action = parts[1]
    cat_key = parts[2]

    if cat_key not in CATEGORIES:
        await cb.answer("Неизвестная категория")
        return

    # Лок проверяем ДО смены категории у пользователя: раньше его здесь не
    # было, и юзер со старой клавиатурой «Быстрых» после истечения Premium
    # спокойно выбирал модель из премиальной категории.
    if category_locked_for(uid, cat_key):
        await cb.answer(
            f"🔒 «{CATEGORIES[cat_key].name}» доступна только для Premium.",
            show_alert=True,
        )
        return

    user_categories[uid] = cat_key

    if action == "view":
        # Переключение вида списка: full — с описаниями (текст),
        # short — компактно с баннером.
        await cb.answer()
        await _show_model_list(cb.message, cat_key, uid, full=(parts[3] == "full"))
        return

    if action == "auto":
        user_specific_model.pop(uid, None)
        save_state()
        await cb.answer("🤖 Auto — бот сам выберет модель")
    elif action == "select":
        mk = parts[3]
        state = _model_overrides().get(mk, {}).get("state")
        if state == "maintenance" and mk in MODELS:
            await cb.answer(
                f"🔧 Модель «{MODELS[mk].name}» сейчас на технических работах.",
                show_alert=True,
            )
            return
        if mk not in MODELS or mk not in category_models(cat_key):
            await cb.answer("Неизвестная модель")
            return
        if mk in ADMIN_ONLY_MODEL_KEYS and not _is_admin(uid):
            await cb.answer("⛔ Эта модель доступна только владельцу бота.",
                            show_alert=True)
            return
        user_specific_model[uid] = mk
        save_state()
        m = MODELS[mk]
        await cb.answer(f"✅ Выбрана: {m.name}")

    await _show_model_list(cb.message, cat_key, uid)


@router.callback_query(F.data.startswith("nvidia_ok:"))
async def cb_nvidia_ok(cb: CallbackQuery) -> None:
    uid = cb.from_user.id
    key = cb.data.split(":", 1)[1]
    if key not in CATEGORIES:
        await cb.answer("Неизвестная категория")
        return

    nvidia_consent[uid] = True
    user_categories[uid] = key
    await cb.answer(f"✅ Категория изменена на: {CATEGORIES[key].name}")
    await _show_model_list(cb.message, key, uid)


@router.callback_query(F.data == "back_start")
async def cb_back_start(cb: CallbackQuery) -> None:
    await cb.answer()
    await _edit_welcome(cb.message, cb.from_user.id)


@router.callback_query(F.data.startswith("menu:"))
async def cb_main_menu(cb: CallbackQuery, bot: Bot) -> None:
    """Кнопки главного инлайн-меню. Каждая делегирует в уже
    существующее действие раздела (как раньше — кнопки Reply-меню)."""
    uid = cb.from_user.id
    action = cb.data.split(":", 1)[1]
    await cb.answer()
    msg = cb.message
    if action == "main":
        await _edit_welcome(msg, uid)
    elif action == "new_chat":
        await _new_chat(msg, uid)
    elif action == "model":
        model_menu_source[uid] = "main"
        await _refresh_model_health()
        await _menu_edit(msg, _category_info_text(user_category(uid), uid),
                         _category_kb(user_category(uid), uid).as_markup(), image="cat_menu")
    elif action == "settings":
        await _edit_settings(msg, uid)
    elif action == "buy":
        await _show_shop(msg, uid)
    elif action == "stats":
        await _show_limit(msg, uid)
    elif action == "help":
        await _show_help(msg, uid)
    elif action == "invite":
        await _send_invite(msg, uid, bot)
    elif action == "bonus":
        await _claim_daily(msg, uid)
    elif action == "health":
        await _refresh_model_health()
        await msg.answer(await _health_message_text(), reply_markup=_model_health_kb())


@router.callback_query(F.data.startswith("auto_switch:"))
async def cb_auto_switch(cb: CallbackQuery, bot: Bot) -> None:
    uid = cb.from_user.id
    choice = cb.data.split(":", 1)[1]
    pending = pending_auto_switch.pop(uid, None)

    if not pending:
        await cb.answer("Это предложение уже неактуально.", show_alert=True)
        return

    if choice == "no":
        await cb.answer("Модель оставлена без изменений.")
        try:
            await cb.message.edit_text(
                "❌ <b>Автоматический режим не включён.</b>\n\n"
                "Выбранная модель сохранена. Повторите попытку позже или выберите другую модель.",
                reply_markup=_menu_btn_kb(),
            )
        except Exception:
            pass
        return

    user_specific_model.pop(uid, None)
    save_state()
    await cb.answer("✅ Auto включён")
    try:
        await cb.message.edit_text(
            "✅ <b>Автоматический режим включён.</b>\n"
            "Повторяю ваш вопрос через доступную модель…"
        )
    except Exception:
        pass

    await _run_ai_turn(
        pending["msg"],
        bot,
        pending["content"],
        category=pending.get("category"),
        _bypass_op=True,
    )


@router.callback_query(F.data == "health:refresh")
async def cb_health_refresh(cb: CallbackQuery) -> None:
    await cb.answer("Проверяю модели…")
    await _refresh_model_health(force=True)
    text = await _health_message_text()
    try:
        await cb.message.edit_text(text, reply_markup=_model_health_kb())
    except Exception:
        await cb.message.answer(text, reply_markup=_model_health_kb())


# ── Кнопки ОП-гейта ────────────────────────────────────────────

@router.callback_query(F.data == "bh_check")
async def cb_bh_check(cb: CallbackQuery) -> None:
    """Кнопка «✅ Проверить» из нашего ОП-поста. Перепроверяет подписку
    через продвинутую интеграцию BotoHub (возвращаются только невыполненные)."""
    uid = cb.from_user.id
    res = await _op_check(uid)
    tasks = res.get("tasks") or []
    if res["completed"] or res["skip"] or not tasks:
        op_pass_date[uid] = _today_iso()
        save_state()
        await cb.answer("Готово! ✅")
        # Убираем кнопки со спонсорского поста, чтобы «Проверить» не жали повторно.
        try:
            await cb.message.edit_reply_markup(reply_markup=None)
        except Exception:
            pass
        pend = pending_ai.pop(uid, None)
        if pend:
            # Авто-ответ на отложенный вопрос (ОП → ответ, без «задайте заново»).
            await _run_ai_turn(pend["msg"], cb.bot, pend["content"],
                               category=pend["category"], _bypass_op=True)
        elif uid not in ADMIN_IDS and not agreement_accepted.get(uid):
            await _send_consent_intro(cb.message)
        else:
            await _send_welcome(cb.message, uid)
    else:
        await cb.answer("Вы подписались не на всех спонсоров 🙏", show_alert=True)
        # Обновляем пост, оставляя только невыполненных спонсоров.
        try:
            await cb.message.edit_reply_markup(reply_markup=_op_tasks_kb(tasks).as_markup())
        except Exception:
            pass


@router.callback_query(F.data.in_({"op_nope", "op_premium"}))
async def cb_op_offer(cb: CallbackQuery) -> None:
    """«Не хочу подписываться» / «Купить премиум» → оффер Premium с вариантом оплаты.
    Окно можно закрыть («⬅️ Закрыть»), кнопки под ОП при этом остаются.
    Цены учитывают активную скидку; при Platega доступна оплата рублями."""
    await cb.answer()
    price_line = (
        f"Оформите за {_disc(PREMIUM_PRICE)} ⭐ или {_disc(PREMIUM_PRICE_RUB)} ₽ — "
        "доступ откроется сразу после оплаты."
        if platega_enabled()
        else f"Оформите за {_disc(PREMIUM_PRICE)} ⭐ — доступ откроется сразу после оплаты."
    )
    await cb.message.answer(
        "💎 <b>Premium — без обязательной подписки и рекламы</b>\n\n"
        "• Никаких спонсоров и рекламы\n"
        "• Доступ к категории «Быстрые»\n"
        "• −20% на списание токенов\n\n"
        + price_line,
        reply_markup=_op_premium_kb().as_markup(),
    )


@router.callback_query(F.data == "op_back")
async def cb_op_back(cb: CallbackQuery) -> None:
    """Закрыть оффер Premium и вернуться к посту со спонсорами (перезапрос)."""
    await cb.answer()
    try:
        await cb.message.delete()
    except Exception:
        pass
    # Если за это время спонсоры выполнены — _run_op_gate вернёт True, ведём далее.
    uid = cb.from_user.id
    if await _run_op_gate(cb.message, uid):
        pend = pending_ai.pop(uid, None)
        if pend:
            await _run_ai_turn(pend["msg"], cb.bot, pend["content"],
                               category=pend["category"], _bypass_op=True)
        elif uid not in ADMIN_IDS and not agreement_accepted.get(uid):
            await _send_consent_intro(cb.message)
        else:
            await _send_welcome(cb.message, uid)


# ══════════════════════════════════════════════════════════════
# КНОПКИ ГЛАВНОГО МЕНЮ (Reply Keyboard)
# ══════════════════════════════════════════════════════════════
# Каждая кнопка делегирует в уже существующее действие раздела.
# Зарегистрированы ВЫШЕ общего текстового хендлера on_text, поэтому
# нажатие на кнопку не уходит в AI как обычный вопрос.

@router.message(F.text == BTN_NEW_CHAT)
async def btn_new_chat(msg: Message) -> None:
    await _new_chat(msg, _uid(msg))


@router.message(F.text == BTN_MODEL)
async def btn_model(msg: Message) -> None:
    await _show_model_menu(msg, _uid(msg))


@router.message(F.text == BTN_SETTINGS)
async def btn_settings(msg: Message) -> None:
    await _send_settings(msg, _uid(msg))


@router.message(F.text == BTN_BUY)
async def btn_buy(msg: Message) -> None:
    await _show_shop(msg, _uid(msg))


@router.message(F.text == BTN_STATS)
async def btn_stats(msg: Message) -> None:
    await _show_limit(msg, _uid(msg))


@router.message(F.text == BTN_HELP)
async def btn_help(msg: Message) -> None:
    await _show_help(msg, _uid(msg))


@router.message(F.text == BTN_INVITE)
async def btn_invite(msg: Message, bot: Bot) -> None:
    await _send_invite(msg, _uid(msg), bot)


@router.message(F.text == BTN_BONUS)
async def btn_bonus(msg: Message) -> None:
    await _claim_daily(msg, _uid(msg))


# ══════════════════════════════════════════════════════════════
# TELEGRAM STARS — ПОКУПКА
# ══════════════════════════════════════════════════════════════

# ── ПРОМОКОДЫ ──────────────────────────────────────────

def _promo_status(info: dict) -> str:
    """'active' | 'expired' | 'exhausted' — состояние промокода."""
    exp = info.get("expires")
    if exp:
        try:
            if date.today() > date.fromisoformat(exp):
                return "expired"
        except Exception:
            return "expired"
    max_uses = int(info.get("max_uses", 0))
    if max_uses and int(info.get("used", 0)) >= max_uses:
        return "exhausted"
    return "active"


@router.message(Command("promo"))
async def cmd_promo(msg: Message) -> None:
    uid = _uid(msg)
    parts = (msg.text or "").split(maxsplit=1)
    if len(parts) < 2:
        await msg.answer("🎟 Введите код так: <code>/promo КОД</code>")
        return
    await _activate_promo(msg, uid, parts[1].strip().upper())


async def _activate_promo(msg: Message, uid: int, code: str) -> None:
    """Общая логика активации: команда /promo и кнопка в настройках."""
    info = promo_codes.get(code)
    if not info or _promo_status(info) != "active":
        await msg.answer("😕 Такой промокод не найден или уже не действует.")
        return
    if uid in info.setdefault("users", []):
        await msg.answer("🙂 Этот промокод вы уже активировали.")
        return
    info["users"].append(uid)
    info["used"] = int(info.get("used", 0)) + 1
    value = int(info.get("value", 0))
    if info.get("kind") == "premium":
        until = grant_premium(uid, value)
        text = (f"🎉 <b>Промокод активирован!</b>\n\n"
                f"💎 Premium на <b>{value}</b> дн. — до {until.strftime('%d.%m.%Y')}")
    else:
        _grant_requests(uid, value)
        text = (f"🎉 <b>Промокод активирован!</b>\n\n"
                f"➕ Начислено: <b>{fmt_tokens(value)}</b> токенов\n"
                f"💰 Баланс: <b>{fmt_tokens(remaining(uid))}</b> токенов")
    save_state()
    await msg.answer(text, reply_markup=_menu_btn_kb())


# ── ПРОБНЫЙ PREMIUM ────────────────────────────────────

@router.callback_query(F.data == "trial:claim")
async def cb_trial_claim(cb: CallbackQuery) -> None:
    uid = cb.from_user.id
    if TRIAL_DAYS <= 0 or uid in trial_used:
        await cb.answer("Пробный период уже был использован 🙂", show_alert=True)
        return
    if premium_active(uid):
        await cb.answer("У вас уже есть Premium 💎", show_alert=True)
        return
    trial_used.add(uid)
    until = grant_premium(uid, TRIAL_DAYS)
    save_state()
    await cb.answer("Premium активирован! 🎉", show_alert=True)
    try:
        await cb.message.edit_text(
            "🎁 <b>Пробный Premium активирован!</b>\n\n"
            f"Действует до: <b>{until.strftime('%d.%m.%Y')}</b>\n\n"
            "⚡ категория «Быстрые», "
            "✨ −20% на списание токенов, 🚫 без рекламы и ОП.",
        )
    except Exception:
        pass


@router.message(Command("buy"))
async def cmd_buy(msg: Message) -> None:
    await _show_shop(msg, _uid(msg))


async def _send_pack_invoice(msg: Message, pack: RequestPack) -> None:
    """Выставляет счёт Telegram Stars на выбранный пакет."""
    await msg.answer_invoice(
        title=f"✨ {fmt_tokens(pack.tokens)} токенов",
        description=f"Пополнение баланса на {fmt_tokens(pack.tokens)} токенов",
        prices=[LabeledPrice(label="XTR", amount=_disc(pack.stars))],
        payload=f"ai_pack:{pack.key}",
        currency="XTR",
    )


@router.callback_query(F.data.startswith("shop:"))
async def cb_shop(cb: CallbackQuery) -> None:
    key = cb.data.split(":", 1)[1]
    pack = REQUEST_PACKS.get(key)
    if not pack:
        await cb.answer("Неизвестный пакет", show_alert=True)
        return
    await cb.answer()
    await _send_pack_invoice(cb.message, pack)


@router.callback_query(F.data == "buy_premium")
async def cb_buy_premium(cb: CallbackQuery) -> None:
    await cb.answer()
    await cb.message.answer_invoice(
        title="💎 Premium-подписка",
        description=(
            f"Premium на {PREMIUM_DAYS} дней: категория «Быстрые», "
            "−20% на списание токенов, без рекламы "
            "и обязательной подписки."
        ),
        prices=[LabeledPrice(label="XTR", amount=_disc(PREMIUM_PRICE))],
        payload=f"premium:{PREMIUM_DAYS}",
        currency="XTR",
    )


# ── Platega: выбор способа оплаты (Stars / рубли) ────────────────

def _kind_rub(kind: str) -> Optional[int]:
    """Цена в рублях для 'premium' или 'pack:<key>'. None — неизвестный kind."""
    if kind == "premium":
        return _disc(PREMIUM_PRICE_RUB)
    if kind.startswith("pack:"):
        pack = REQUEST_PACKS.get(kind.split(":", 1)[1])
        return _disc(pack.rub) if pack else None
    if kind.startswith("topup:"):
        try:
            rub = int(kind.split(":", 1)[1])
            return rub if TOPUP_MIN_RUB <= rub <= TOPUP_MAX_RUB else None
        except ValueError:
            return None
    return None


def _kind_title(kind: str) -> str:
    if kind == "premium":
        return f"Premium на {PREMIUM_DAYS} дней"
    if kind.startswith("pack:"):
        pack = REQUEST_PACKS.get(kind.split(":", 1)[1])
        if pack:
            return f"{fmt_tokens(pack.tokens)} токенов"
    if kind.startswith("topup:"):
        rub = _kind_rub(kind) or 0
        return f"Пополнение на {fmt_tokens(rub * TOKENS_PER_RUB)} токенов"
    return "Покупка"


def _pay_methods_kb(kind: str) -> InlineKeyboardBuilder:
    """Меню способов оплаты для выбранной покупки (kind='premium'|'pack:<key>')."""
    b = InlineKeyboardBuilder()
    if kind == "premium":
        b.row(InlineKeyboardButton(text=f"⭐ Telegram Stars — {_disc(PREMIUM_PRICE)}", callback_data="buy_premium"))
    elif kind.startswith("pack:"):
        key = kind.split(":", 1)[1]
        b.row(InlineKeyboardButton(text=f"⭐ Telegram Stars — {_disc(REQUEST_PACKS[key].stars)}", callback_data=f"shop:{key}"))
    rub = _kind_rub(kind)
    if platega_enabled() and rub:
        for mk, m in PLATEGA_METHODS.items():
            b.row(InlineKeyboardButton(text=f"{m['emoji']} {m['name']} — {rub}₽", callback_data=f"pg:{mk}:{kind}"))
    b.row(InlineKeyboardButton(text="⬅️ Назад", callback_data="shop_back"))
    return b


@router.callback_query(F.data.startswith("pmenu:"))
async def cb_pay_menu(cb: CallbackQuery) -> None:
    kind = cb.data.split(":", 1)[1]   # "premium" | "pack:<key>"
    if _kind_rub(kind) is None:
        await cb.answer("Неизвестная позиция", show_alert=True)
        return
    await cb.answer()
    try:
        await cb.message.edit_text(
            f"💳 <b>Оплата: {_kind_title(kind)}</b>\n\nВыберите способ оплаты:",
            reply_markup=_pay_methods_kb(kind).as_markup(),
        )
    except Exception:
        pass


@router.callback_query(F.data == "shop_back")
async def cb_shop_back(cb: CallbackQuery) -> None:
    uid = cb.from_user.id
    await cb.answer()
    try:
        await cb.message.edit_text(_shop_text(uid), reply_markup=_shop_kb(uid).as_markup())
    except Exception:
        pass


@router.callback_query(F.data.startswith("pg:"))
async def cb_pg_pay(cb: CallbackQuery) -> None:
    """Создаёт платёж Platega и присылает ссылку + кнопку «Я оплатил»."""
    parts = cb.data.split(":")
    method_key = parts[1]
    kind = ":".join(parts[2:])       # "premium" | "pack:<key>"
    uid = cb.from_user.id
    method = PLATEGA_METHODS.get(method_key)
    rub = _kind_rub(kind)
    if not method or rub is None:
        await cb.answer("Способ недоступен", show_alert=True)
        return
    if uid in _platega_creating:
        await cb.answer("Счёт уже создаётся — секунду…", show_alert=True)
        return
    _platega_creating.add(uid)
    try:
        await _cb_pg_pay_inner(cb, uid, method, kind, rub)
    finally:
        _platega_creating.discard(uid)


async def _cb_pg_pay_inner(cb: CallbackQuery, uid: int,
                           method: dict, kind: str, rub: int) -> None:
    await cb.answer("Создаю счёт…")

    user_name = f"@{cb.from_user.username}" if cb.from_user.username else cb.from_user.full_name
    res = await _platega_create(
        uid, rub, description=_kind_title(kind),
        payload=f"{kind}|{uid}", method_code=method["code"], user_name=user_name,
    )
    if not res:
        try:
            await cb.message.edit_text(
                "😔 Не удалось создать счёт. Попробуйте другой способ оплаты "
                "или Telegram Stars.",
                reply_markup=_pay_methods_kb(kind).as_markup(),
            )
        except Exception:
            pass
        return

    tx_id = res["transactionId"]
    platega_tx[tx_id] = {"uid": uid, "kind": kind, "amount_rub": rub, "credited": False}
    save_state()

    b = InlineKeyboardBuilder()
    b.row(InlineKeyboardButton(text=f"{method['emoji']} Оплатить {rub}₽", url=res["redirect"]))
    b.row(InlineKeyboardButton(text="✅ Я оплатил", callback_data=f"pgcheck:{tx_id}"))
    b.row(InlineKeyboardButton(text="⬅️ Назад", callback_data="shop_back"))
    try:
        await cb.message.edit_text(
            f"🧾 <b>Счёт создан: {_kind_title(kind)}</b>\n\n"
            f"💰 К оплате: <b>{rub} ₽</b> ({method['name']})\n"
            "⏳ Ссылка действует ограниченное время.\n\n"
            "1️⃣ Нажмите «Оплатить» и завершите платёж\n"
            "2️⃣ Вернитесь и нажмите «✅ Я оплатил»",
            reply_markup=b.as_markup(),
        )
    except Exception:
        pass


async def _credit_platega(msg: Message, uid: int, kind: str, amount_rub: int = 0) -> None:
    """Начисляет покупку после подтверждения оплаты Platega.

    Все ветки обязаны оставить состояние на диске: флаг credited лёг ДО начисления
    (см. cb_pg_check), поэтому потеря начисления = «деньги взяты, товар не выдан»
    без возможности повторить. Гарантия — save_state_now в finally: две ветки ниже
    (неизвестный пакет, нулевая сумма) не начисляли ничего и раньше не сохраняли
    даже запись в журнале покупок, хотя это именно тот случай, который нужно
    разбирать руками.
    """
    try:
        await _credit_platega_inner(msg, uid, kind, amount_rub)
    finally:
        # Только если ветка не сохранилась сама: успешные ветки пишут сразу после
        # начисления (важно — до долгого _run_ai_turn), и повторная полная
        # перезапись 27 таблиц там ни к чему.
        if _state_dirty:
            save_state_now()


async def _credit_platega_inner(msg: Message, uid: int, kind: str, amount_rub: int = 0) -> None:
    _log_purchase(uid, kind, _kind_title(kind), amount_rub, "RUB")
    await _notify_admins_purchase(msg.bot, uid, _kind_title(kind), amount_rub, "RUB")
    if kind == "premium":
        until = grant_premium(uid, PREMIUM_DAYS)
        # save_state_now, а не save_state: флаг rec["credited"] уже лёг на диск
        # ДО начисления (см. cb_pg_check). Если начисление останется только в
        # памяти и процесс упадёт в окне автосейва, после старта поднимется
        # «платёж обработан» с нулевым балансом, а идемпотентность не даст
        # начислить повторно — деньги взяты, товар не выдан.
        save_state_now()
        await msg.answer(
            "💎 <b>Premium активирован!</b>\n\n"
            f"Действует до: <b>{until.strftime('%d.%m.%Y')}</b>\n\n"
            "⚡ категория «Быстрые», "
            "✨ −20% на списание токенов, 🚫 без рекламы и ОП.",
            reply_markup=_menu_btn_kb(),
        )
        pend = pending_ai.pop(uid, None)
        if pend:
            await _run_ai_turn(pend["msg"], msg.bot, pend["content"],
                               category=pend["category"], _bypass_op=True)
    elif kind.startswith("pack:"):
        pack = REQUEST_PACKS.get(kind.split(":", 1)[1])
        if pack:
            _grant_requests(uid, pack.tokens)
            save_state_now()  # см. комментарий выше про идемпотентность
            await msg.answer(
                "✅ <b>Оплата прошла!</b>\n\n"
                f"➕ Начислено: <b>{fmt_tokens(pack.tokens)}</b> токенов\n"
                f"💰 Баланс: <b>{fmt_tokens(remaining(uid))}</b> токенов\n\n"
                "Просто пишите — AI ответит!",
                reply_markup=_menu_btn_kb(),
            )
    elif kind.startswith("topup:"):
        tokens = (amount_rub or 0) * TOKENS_PER_RUB
        if tokens > 0:
            _grant_requests(uid, tokens)
            save_state_now()  # см. комментарий выше про идемпотентность
            await msg.answer(
                "✅ <b>Оплата прошла!</b>\n\n"
                f"➕ Начислено: <b>{fmt_tokens(tokens)}</b> токенов\n"
                f"💰 Баланс: <b>{fmt_tokens(remaining(uid))}</b> токенов\n\n"
                "Просто пишите — AI ответит!",
                reply_markup=_menu_btn_kb(),
            )


@router.callback_query(F.data.startswith("pgcheck:"))
async def cb_pg_check(cb: CallbackQuery) -> None:
    """Проверяет статус платежа Platega и начисляет покупку (идемпотентно)."""
    tx_id = cb.data.split(":", 1)[1]
    rec = platega_tx.get(tx_id)
    if not rec or rec.get("uid") != cb.from_user.id:
        await cb.answer("Счёт не найден", show_alert=True)
        return
    if rec.get("credited"):
        await cb.answer("Эта покупка уже зачислена ✅", show_alert=True)
        return
    # Защита от гонки: при двойном нажатии «Я оплатил» aiogram запускает
    # хендлеры конкурентно. Блокируем tx_id в памяти ДО await, чтобы второй
    # клик не проскочил проверку credited и не начислил оплату повторно.
    # Набор внутрипамятный (не персистится) — не «залипнет» после рестарта.
    if tx_id in _platega_checking:
        await cb.answer("Проверяю оплату — секунду…", show_alert=True)
        return
    _platega_checking.add(tx_id)
    try:
        data = await _platega_status(tx_id)
        status = (data or {}).get("status")
        if status == "CONFIRMED":
            if rec.get("credited"):
                await cb.answer("Эта покупка уже зачислена ✅", show_alert=True)
                return
            # Сверка суммы: не начисляем, если провайдер сообщил оплату меньше
            # ожидаемой (защита от подмены суммы на стороне платёжной страницы).
            expected = rec.get("amount_rub", 0)
            paid = _platega_paid_amount(data or {})
            if paid is not None and expected and paid + 0.01 < expected:
                logging.warning(
                    f"Platega tx {tx_id}: оплачено {paid} < ожидалось {expected}, uid={cb.from_user.id}"
                )
                await cb.answer(
                    "Сумма оплаты не совпала с счётом. Свяжитесь с поддержкой.",
                    show_alert=True,
                )
                return
            rec["credited"] = True
            save_state_now()  # фиксируем идемпотентность до начисления
            await cb.answer("Оплата подтверждена! ✅", show_alert=True)
            try:
                await cb.message.edit_reply_markup(reply_markup=None)
            except Exception:
                pass
            await _credit_platega(cb.message, cb.from_user.id, rec["kind"], rec.get("amount_rub", 0))
        elif status in ("CANCELED", "CHARGEBACKED"):
            await cb.answer("Платёж отменён. Создайте счёт заново.", show_alert=True)
        else:
            await cb.answer(
                "Оплата пока не поступила. Если вы только что оплатили — "
                "подождите минуту и нажмите «✅ Я оплатил» ещё раз.",
                show_alert=True,
            )
    finally:
        _platega_checking.discard(tx_id)


# ── ЖУРНАЛ ПРОДАЖ + УВЕДОМЛЕНИЯ АДМИНАМ ────────────────────

def _log_purchase(uid: int, kind: str, title: str, amount: int, currency: str) -> None:
    """Пишет покупку в журнал продаж (статистика админки + экспорт CSV)."""
    purchases.append({
        "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "uid": uid, "kind": kind, "title": title,
        "amount": int(amount), "currency": currency,
    })
    # Журнал не растёт бесконечно: save_state_now перезаписывает таблицу целиком
    # и синхронно, то есть каждая запись подтормаживает всех пользователей на
    # время записи ВСЕХ покупок за историю бота. 5000 последних хватает и для
    # статистики, и для выгрузки, а старое всё равно лежит в ежедневных бэкапах.
    if len(purchases) > MAX_PURCHASES:
        del purchases[:len(purchases) - MAX_PURCHASES]
    # save_state_now здесь НЕ вызываем: вызывающий всё равно пишет на диск сразу
    # после начисления, и раньше на одну покупку приходилось две полные
    # перезаписи 27 таблиц. Одна запись после начисления вместо двух — это ещё и
    # цельнее: журнал и выданный товар попадают на диск вместе, а не порознь.
    save_state()


async def _notify_admins_purchase(bot: Bot, uid: int, title: str, amount: int, currency: str) -> None:
    """Сообщает всем админам о новой покупке (best-effort)."""
    s = user_stats.get(uid, {})
    uname = f"@{s['username']}" if s.get("username") else s.get("full_name", str(uid))
    cur = "⭐" if currency == "XTR" else "₽"
    for admin_id in ADMIN_IDS:
        try:
            await bot.send_message(
                admin_id,
                f"💰 <b>Новая покупка!</b>\n\n"
                f"👤 {html.quote(str(uname))} (<code>{uid}</code>)\n"
                f"🛒 {title}\n"
                f"💵 {amount} {cur}",
            )
        except Exception:
            pass


@router.pre_checkout_query()
async def pre_checkout(query: PreCheckoutQuery) -> None:
    # Деньги ещё НЕ списаны — здесь можно отказать без потерь. Без этой
    # проверки не принявший соглашение пользователь платил через старые
    # кнопки, а successful_payment резался гвардом: деньги взяты, товар
    # не выдан, повторная доставка упиралась в тот же гвард.
    uid = query.from_user.id
    if uid not in ADMIN_IDS and not agreement_accepted.get(uid):
        await query.answer(
            ok=False,
            error_message="Сначала отправьте /start и примите соглашение.",
        )
        return
    await query.answer(ok=True)


@router.message(F.successful_payment)
async def on_payment(msg: Message) -> None:
    uid = _uid(msg)
    payload = msg.successful_payment.invoice_payload or ""

    # Идемпотентность: повторная доставка того же платежа не начисляется дважды.
    charge_id = msg.successful_payment.telegram_payment_charge_id or ""
    if charge_id:
        if charge_id in stars_charges:
            logging.warning(f"Stars: повторный successful_payment {charge_id} от {uid} — пропущен")
            return
        stars_charges[charge_id] = {
            "uid": uid,
            "stars": int(msg.successful_payment.total_amount or 0),
            "payload": payload,
            "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "refunded": False,
        }

    # Покупка Premium-подписки
    if payload.startswith("premium"):
        parts = payload.split(":")
        days = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else PREMIUM_DAYS
        until = grant_premium(uid, days)  # Premium снимает ОП-гейт (см. _op_required)
        stars_paid = msg.successful_payment.total_amount
        _log_purchase(uid, "premium", f"Premium {days} дн.", stars_paid, "XTR")
        await _notify_admins_purchase(msg.bot, uid, f"Premium {days} дн.", stars_paid, "XTR")
        # Немедленная запись: вместе с начислением на диск обязан лечь и леджер
        # stars_charges, иначе после падения в окне автосейва пропадут и токены,
        # и защита от двойного начисления.
        save_state_now()
        await msg.answer(
            "💎 <b>Premium активирован!</b>\n\n"
            f"Действует до: <b>{until.strftime('%d.%m.%Y')}</b>\n\n"
            "Теперь доступно:\n"
            "⚡ категория «Быстрые»\n"
            "✨ −20% на списание токенов\n"
            "🚫 без рекламы и обязательной подписки",
            reply_markup=_menu_btn_kb(),
        )
        # Если пользователь купил Premium, упёршись в ОП, — сразу отвечаем на
        # отложенный вопрос (Premium снял гейт).
        pend = pending_ai.pop(uid, None)
        if pend:
            await _run_ai_turn(pend["msg"], msg.bot, pend["content"],
                               category=pend["category"], _bypass_op=True)
            return
        # Иначе, если ещё не принял соглашение — ведём к согласию.
        if uid not in ADMIN_IDS and not agreement_accepted.get(uid):
            await _send_consent_intro(msg)
        return

    # Пополнение произвольной суммой в Stars
    if payload == "topup_stars":
        stars_paid = msg.successful_payment.total_amount
        tokens = stars_paid * TOKENS_PER_STAR
        u = _get_usage(uid)
        u["bought"] += tokens
        _log_purchase(uid, "topup_stars", f"Пополнение {fmt_tokens(tokens)} ткн", stars_paid, "XTR")
        await _notify_admins_purchase(msg.bot, uid, f"Пополнение {fmt_tokens(tokens)} ткн", stars_paid, "XTR")
        save_state_now()  # см. комментарий в ветке premium
        await msg.answer(
            "✅ <b>Оплата прошла!</b>\n\n"
            f"➕ Начислено: <b>{fmt_tokens(tokens)}</b> токенов\n"
            f"💰 Баланс: <b>{fmt_tokens(remaining(uid))}</b> токенов\n\n"
            "Просто пишите — AI ответит!",
            reply_markup=_menu_btn_kb(),
        )
        return

    # Покупка пакета токенов
    pack = _pack_from_payload(payload)
    if pack is None:
        # Пакет удалили/переименовали между инвойсом и оплатой. Деньги УЖЕ
        # списаны: ничего не начисляем молча (раньше уехал бы дефолтный p50),
        # фиксируем платёж в леджере и зовём админа разбираться вручную.
        logging.error(
            f"Stars: оплачен неизвестный пакет payload={payload!r} uid={uid}, "
            f"stars={msg.successful_payment.total_amount}"
        )
        await _notify_admins_purchase(
            msg.bot, uid,
            f"⚠️ НЕИЗВЕСТНЫЙ пакет ({payload}) — начисли вручную!",
            msg.successful_payment.total_amount, "XTR",
        )
        save_state_now()  # леджер stars_charges уже должен лечь на диск
        await msg.answer(
            "⚠️ <b>Оплата получена, но товар не распознан.</b>\n\n"
            "Мы уже уведомлены и начислим токены вручную. "
            f"Поддержка: {SUPPORT_USERNAME}",
        )
        return
    u = _get_usage(uid)
    u["bought"] += pack.tokens
    stars_paid = msg.successful_payment.total_amount
    _log_purchase(uid, f"pack:{pack.key}", f"{fmt_tokens(pack.tokens)} токенов", stars_paid, "XTR")
    await _notify_admins_purchase(msg.bot, uid, f"{fmt_tokens(pack.tokens)} токенов", stars_paid, "XTR")
    save_state_now()  # см. комментарий в ветке premium
    left = remaining(uid)
    await msg.answer(
        "✅ <b>Оплата прошла!</b>\n\n"
        f"➕ Начислено: <b>{fmt_tokens(pack.tokens)}</b> токенов\n"
        f"💰 Баланс: <b>{fmt_tokens(left)}</b> токенов\n\n"
        "Просто пишите — AI ответит!",
        reply_markup=_menu_btn_kb(),
    )


# ══════════════════════════════════════════════════════════════
# ПРОВЕРКА ЛИМИТА
# ══════════════════════════════════════════════════════════════

async def _check_limit(msg: Message, uid: int) -> bool:
    if not can_use(uid):
        b = InlineKeyboardBuilder()
        b.row(InlineKeyboardButton(text="💳 Пополнить баланс", callback_data="buy_pack"))
        await msg.answer(
            "⚠️ <b>Токены закончились!</b>\n\n"
            "Пополните баланс токенов, чтобы продолжить 👇",
            reply_markup=b.as_markup(),
        )
        return False
    return True


@router.callback_query(F.data == "buy_pack")
async def cb_buy_pack(cb: CallbackQuery) -> None:
    await cb.answer()
    await _show_shop(cb.message, cb.from_user.id)


async def _send_token_receipt(msg: Message, tin: int, tout: int, spent: int,
                              coef: float = 1.0, admin_free: bool = False) -> None:
    """Мини-квитанция под ответом: списание + остаток одной строкой.
    coef — тариф модели (показываем 💎×N, если дороже обычного).
    admin_free=True — сумма показана для проверки, но с баланса не снята."""
    uid = _uid(msg)
    b = InlineKeyboardBuilder()
    b.row(InlineKeyboardButton(
        text="📊 Детали",
        callback_data=f"tok:{tin}:{tout}:{spent}:{coef:g}",
    ))
    badge = f" 💎×{coef:g}" if coef > 1 else ""
    if admin_free:
        label = f"🛡 −{fmt_tokens(spent)}{badge} · тест, не списано"
    else:
        label = f"✨ −{fmt_tokens(spent)}{badge} · 💰 {fmt_tokens(remaining(uid))}"
    try:
        await msg.answer(label, reply_markup=b.as_markup())
    except Exception:
        pass


@router.callback_query(F.data.startswith("tok:"))
async def cb_token_receipt(cb: CallbackQuery) -> None:
    """Детализация списания: input / output / тариф / итог."""
    try:
        parts = cb.data.split(":")
        tin, tout, spent = parts[1], parts[2], parts[3]
        coef = float(parts[4]) if len(parts) > 4 else 1.0   # старые кнопки — без коэф.
        lines = [
            f"⬆️ Вопрос + контекст: {fmt_tokens(int(tin))}",
            f"⬇️ Ответ: {fmt_tokens(int(tout))}",
        ]
        if coef > 1:
            lines.append(f"💎 Тариф модели: ×{coef:g}")
        lines += [
            f"💸 Списано: {fmt_tokens(int(spent))}",
            f"💰 Баланс: {fmt_tokens(remaining(cb.from_user.id))}",
        ]
        await cb.answer("\n".join(lines), show_alert=True)
    except Exception:
        await cb.answer()


@router.callback_query(F.data == "topup:stars")
async def cb_topup_stars(cb: CallbackQuery) -> None:
    user_input_state[cb.from_user.id] = "topup_stars"
    await cb.answer()
    await cb.message.answer(
        "💫 <b>Пополнение в Telegram Stars</b>\n\n"
        f"Курс: 1 ⭐ = <b>{TOKENS_PER_STAR}</b> токенов\n"
        f"Введите сумму от {TOPUP_MIN_STARS} до {TOPUP_MAX_STARS} ⭐ одним числом.\n\n"
        "Отмена — /cancel",
    )


@router.callback_query(F.data == "topup:rub")
async def cb_topup_rub(cb: CallbackQuery) -> None:
    if not platega_enabled():
        await cb.answer("Оплата рублями временно недоступна", show_alert=True)
        return
    user_input_state[cb.from_user.id] = "topup_rub"
    await cb.answer()
    await cb.message.answer(
        "💳 <b>Пополнение в рублях</b>\n\n"
        f"Курс: 1 ₽ = <b>{TOKENS_PER_RUB}</b> токенов\n"
        f"Введите сумму от {TOPUP_MIN_RUB} до {TOPUP_MAX_RUB} ₽ одним числом.\n\n"
        "Отмена — /cancel",
    )


# Список префиксов ошибок. БОЛЬШЕ НЕ УЧАСТВУЕТ в решении «ошибка или нет» —
# оставлен как справка о том, с чего начинаются наши тексты ошибок.
# Почему отказались: первый символ ответа задаёт модель, а её поведением
# управляет пользователь. Просьба «начинай ответ с ⚠️» превращала каждый ответ
# в «ошибку»: токены не списывались, бесплатный ответ не сгорал, текст приходил
# целиком. Теперь ошибки помечены типом AIError (см. _is_error_answer), и добавляя
# новую ветку ошибки в ask_ai, оборачивай её в AIError(...), а не следи за эмодзи.
_ERROR_PREFIXES = ("😔", "⏱️", "❌", "🔑", "🔧", "⚠️")


def _is_error_answer(answer: str) -> bool:
    """Ошибка ли это. Решает ТИП (AIError), а не первый символ текста.

    Проверка по префиксу оставлена только как страховка для ошибок, которые
    собираются вне ask_ai. Полагаться на неё нельзя: первый символ приходит от
    модели, а значит управляем пользователем через промпт («начинай с ⚠️») — так
    любой ответ объявлялся ошибкой и выдавался бесплатно.
    """
    # Единственный источник — ask_ai, а он теперь возвращает AIError для ошибок
    # и обычную строку для ответа модели. Ответ, начинающийся с «⚠️», ошибкой
    # больше НЕ считается: именно это и было лазейкой.
    return isinstance(answer, AIError)


def _auto_switch_confirm_kb() -> InlineKeyboardMarkup:
    """Подтверждение перехода с недоступной модели на Auto."""
    b = InlineKeyboardBuilder()
    b.row(
        InlineKeyboardButton(
            text="✅ Да, включить Auto",
            callback_data="auto_switch:yes",
            style="success",
        ),
        InlineKeyboardButton(
            text="❌ Нет",
            callback_data="auto_switch:no",
            style="danger",
        ),
    )
    return b.as_markup()


async def _run_ai_turn(msg: Message, bot: Bot, content, *, category: Optional[str] = None,
                        thinking_label: str = "🧠 <i>Думаю...</i>",
                        _bypass_op: bool = False) -> None:
    """Общий цикл: ОП-гейт -> typing-индикатор -> запрос к AI -> отправка ответа.

    Единственная точка ответа для всех типов ввода (текст/фото/голос/видео),
    поэтому ОП-гейт стоит здесь. Если ОП нужен — придерживаем ответ, сохраняем
    запрос в pending_ai и показываем пост со спонсорами; после «✅ Проверить»
    cb_bh_check переигрывает этот вызов с _bypass_op=True.
    """
    uid = _uid(msg)
    chat_id = msg.chat.id

    if not _bypass_op and _op_required(uid):
        # Показываем спонсоров. Если BotoHub вернул спонсоров (gate=False) —
        # придерживаем ответ до «✅ Проверить». Если спонсоров нет / BotoHub
        # недоступен (gate=True, fail-open) — гейт уже отмечен пройденным,
        # отвечаем сразу, не оставляя юзера без ответа.
        if not await _run_op_gate(msg, uid):
            pending_ai[uid] = {"msg": msg, "content": content, "category": category}
            return

    # Анти-флуд: не даём одному пользователю запускать несколько генераций
    # параллельно. Пока идёт ответ — новое сообщение мягко отклоняем.
    if uid in processing_users:
        try:
            await msg.answer("⏳ Секунду — я ещё дописываю предыдущий ответ.")
        except Exception as e:
            logging.debug(f"антифлуд: отказ не доставлен (uid={uid}): {e}")
        return

    # Кулдаун между вопросами одного человека. Анти-флуд выше запрещает только
    # ПАРАЛЛЕЛЬНЫЕ генерации, а последовательные шли без ограничений: скрипт мог
    # молотить вопросы один за другим и выжигать общие ключи провайдеров для всех
    # остальных. Админов не тормозим.
    now_ts = time.monotonic()
    if uid not in ADMIN_IDS and USER_COOLDOWN_SEC > 0:
        prev_ts = last_ai_request_at.get(uid)
        if prev_ts is not None:
            wait_left = USER_COOLDOWN_SEC - (now_ts - prev_ts)
            if wait_left > 0:
                try:
                    await msg.answer(
                        f"⏳ Слишком часто. Подождите {wait_left:.0f} с и спросите снова."
                    )
                except Exception as e:
                    logging.debug(f"кулдаун: отказ не доставлен (uid={uid}): {e}")
                return
    # Штамп ставим и здесь, и в finally. Здесь — чтобы отсечь два сообщения,
    # пришедших в один момент; в finally — чтобы пауза считалась от КОНЦА ответа.
    # Только начального штампа не хватало: ответ идёт 10-30 с, кулдаун 3 с уже
    # истёк к моменту его получения, и последовательные вопросы шли без паузы —
    # то есть защита общих ключей не работала вовсе.
    last_ai_request_at[uid] = now_ts

    processing_users.add(uid)
    try:
        async with TypingIndicator(bot, chat_id):
            status_msg = await msg.answer(thinking_label)
            try:
                # Дедлайн на весь перебор моделей: внутренний REQUEST_TIMEOUT
                # ограничивает ОДИН запрос, а их может быть 8 подряд с ретраями.
                answer = await asyncio.wait_for(
                    ask_ai(uid, content, status_msg=status_msg, category=category),
                    timeout=AI_TURN_TIMEOUT,
                )
            except asyncio.TimeoutError:
                # ask_ai отменён на середине, поэтому его собственный _rollback
                # не выполнился — убираем вопрос из истории здесь, иначе он
                # уедет в следующий запрос как «уже заданный».
                _rollback(uid)
                last_input_tokens.pop(uid, None)
                last_token_coef.pop(uid, None)
                try:
                    await status_msg.delete()
                except Exception as e:
                    logging.debug(f"timeout: не удалось убрать статус: {e}")
                logging.warning(f"AI turn timeout {AI_TURN_TIMEOUT}s (uid={uid})")
                await msg.answer(
                    f"⏱️ <b>Модели не ответили за {AI_TURN_TIMEOUT} с.</b>\n\n"
                    "Обычно это перегрузка на стороне провайдера. "
                    "Попробуйте переспросить или выберите другую категорию (/model).\n\n"
                    "Токены за этот запрос не списаны.",
                )
                return
            except SelectedModelUnavailable as exc:
                last_input_tokens.pop(uid, None)
                last_token_coef.pop(uid, None)
                try:
                    await status_msg.delete()
                except Exception:
                    pass
                model = MODELS.get(exc.model_key)
                model_name = model.name if model else exc.model_key
                pending_auto_switch[uid] = {
                    "msg": msg,
                    "content": content,
                    "category": category,
                    "model_key": exc.model_key,
                }
                await msg.answer(
                    f"🔴 <b>Модель «{html.quote(model_name)}» сейчас недоступна.</b>\n\n"
                    "Желаете переключиться на автоматический режим?",
                    reply_markup=_auto_switch_confirm_kb(),
                )
                return
            is_error = _is_error_answer(answer)
            tin = last_input_tokens.pop(uid, 0)
            coef = last_token_coef.pop(uid, 1.0)
            tout = _estimate_tokens(answer) if not is_error else 0
            spent = _calc_spend(uid, tin, tout, coef) if not is_error else 0
            # Админам квитанция показывается (проверка расценок), но токены
            # с баланса не снимаются.
            if spent and uid not in ADMIN_IDS:
                _spend_tokens(uid, spent)
                save_state()
            try:
                await status_msg.delete()
            except Exception:
                pass

        bot_msg_id = await send_response(msg, answer, uid=uid)
        if bot_msg_id is not None:
            _remember_bot_msg(chat_id, msg.message_id, bot_msg_id)

        # Квитанция о списании токенов (админу — с пометкой, без списания)
        if spent:
            await _send_token_receipt(msg, tin, tout, spent, coef=coef,
                                      admin_free=uid in ADMIN_IDS)
            await _maybe_activate_referral(bot, uid)

        # Списываем бесплатный ответ только за реально отданный ответ и только пока
        # юзер в бесплатной фазе (не прошёл ОП сегодня, не premium/админ). Иначе
        # следующий запрос упрётся в ОП.
        if not is_error and uid not in ADMIN_IDS and not premium_active(uid) \
                and op_pass_date.get(uid) != _today_iso() \
                and op_free_used.get(uid, 0) < OP_FREE_ANSWERS:
            op_free_used[uid] = op_free_used.get(uid, 0) + 1
            save_state()

        await _botohub_maybe_show_ad(uid)
    finally:
        processing_users.discard(uid)
        # Пауза отсчитывается от конца ответа, поэтому штамп обновляем здесь —
        # в finally, чтобы он ставился и при ошибке, и при таймауте, и при
        # раннем return из блока try (иначе после сбоя кулдауна не будет).
        if uid not in ADMIN_IDS:
            last_ai_request_at[uid] = time.monotonic()



# ══════════════════════════════════════════════════════════════
# ОБРАБОТКА — ТЕКСТ
# ══════════════════════════════════════════════════════════════

@router.message(F.text & ~F.text.startswith("/"))
async def on_text(msg: Message, bot: Bot) -> None:
    if not msg.text:
        return
    uid = _uid(msg)

    state = admin_state.get(uid)
    if state and _is_admin(uid):
        if state == "waiting_broadcast":
            await _preview_broadcast(msg)
            return
        if state == "waiting_user_search":
            admin_state.pop(uid, None)
            await _admin_handle_search(msg)
            return
        if state.startswith("waiting_dm:"):
            admin_state.pop(uid, None)
            await _admin_handle_dm(msg, int(state.split(":")[1]))
            return
        if state == "waiting_promo":
            admin_state.pop(uid, None)
            await _admin_handle_promo_create(msg)
            return
        if state == "waiting_give_all":
            admin_state.pop(uid, None)
            await _admin_handle_give_all(msg)
            return
        if state == "waiting_selfedit":
            admin_state.pop(uid, None)
            instruction = (msg.text or "").strip()
            if not instruction:
                await msg.answer("😕 Пустая инструкция. Откройте админку и попробуйте снова.")
                return
            await _do_selfedit(msg, instruction, uid)
            return
        if state == "waiting_sale":
            admin_state.pop(uid, None)
            await _admin_handle_sale(msg)
            return
        if state == "waiting_model_add":
            await _admin_model_add_step(msg)
            return
        if state.startswith("model_desc:"):
            await _admin_model_description_step(msg)
            return

    if user_input_state.get(uid) == "promo_code":
        user_input_state.pop(uid, None)
        code = (msg.text or "").strip().upper()
        if not code:
            await msg.answer("😕 Пустой код. Откройте настройки и попробуйте ещё раз.")
            return
        await _activate_promo(msg, uid, code)
        return

    if user_input_state.get(uid) == "topup_stars":
        user_input_state.pop(uid, None)
        raw = (msg.text or "").strip()
        try:
            stars = int(raw)
        except ValueError:
            await msg.answer("😕 Нужно одно число — сумма в ⭐. Откройте магазин и попробуйте ещё раз.")
            return
        if not (TOPUP_MIN_STARS <= stars <= TOPUP_MAX_STARS):
            await msg.answer(f"😕 Сумма должна быть от {TOPUP_MIN_STARS} до {TOPUP_MAX_STARS} ⭐.")
            return
        tokens = stars * TOKENS_PER_STAR
        await msg.answer_invoice(
            title=f"✨ {fmt_tokens(tokens)} токенов",
            description=f"Пополнение баланса на {fmt_tokens(tokens)} токенов",
            prices=[LabeledPrice(label="XTR", amount=stars)],
            payload="topup_stars",
            currency="XTR",
        )
        return

    if user_input_state.get(uid) == "topup_rub":
        user_input_state.pop(uid, None)
        raw = (msg.text or "").strip()
        try:
            rub = int(raw)
        except ValueError:
            await msg.answer("😕 Нужно одно число — сумма в ₽. Откройте магазин и попробуйте ещё раз.")
            return
        if not (TOPUP_MIN_RUB <= rub <= TOPUP_MAX_RUB):
            await msg.answer(f"😕 Сумма должна быть от {TOPUP_MIN_RUB} до {TOPUP_MAX_RUB} ₽.")
            return
        kind = f"topup:{rub}"
        await msg.answer(
            f"💳 <b>Оплата: {_kind_title(kind)}</b>\n\nВыберите способ оплаты:",
            reply_markup=_pay_methods_kb(kind).as_markup(),
        )
        return

    if user_input_state.get(uid) == "custom_prompt":
        user_input_state.pop(uid, None)
        user_custom_prompt[uid] = (msg.text or "").strip()[:CUSTOM_PROMPT_MAX_LEN]
        save_state()
        await msg.answer(
            "📝 <b>Промпт сохранён!</b>\n\nТеперь бот будет учитывать его в ответах.",
            reply_markup=_menu_btn_kb(),
        )
        return

    _track_user(uid, msg.from_user)
    if not await _check_limit(msg, uid):
        return

    await _run_ai_turn(msg, bot, msg.text)


async def _preview_broadcast(msg: Message) -> None:
    """Превью рассылки: показываем пост админу и ждём подтверждения."""
    uid = _uid(msg)
    admin_state.pop(uid, None)
    text = msg.text or ""

    # Конвертируем Markdown → HTML для превью (админ видит, как будет выглядеть)
    html_text = _md_to_html(text)

    # Пробуем показать с HTML, при ошибке — показываем как plain text
    preview_ok = False
    try:
        await msg.answer(html_text, parse_mode=ParseMode.HTML)
        preview_ok = True
    except Exception:
        try:
            await msg.answer(text, parse_mode=None)
        except Exception:
            pass

    if not preview_ok:
        admin_state[uid] = "waiting_broadcast"
        await msg.answer(
            "⚠️ <b>Ошибка разметки</b> — Telegram не смог разобрать текст.\n"
            "Проверьте Markdown/HTML-теги и пришлите текст ещё раз "
            "(или нажмите «Отмена»)."
        )
        return

    admin_pending_broadcast[uid] = text
    b = InlineKeyboardBuilder()
    b.row(
        InlineKeyboardButton(text="✅ Отправить всем", callback_data="admin:broadcast_send"),
        InlineKeyboardButton(text="❌ Отмена", callback_data="admin:broadcast_cancel"),
    )
    await msg.answer(
        f"👆 Выше — превью: так пост увидят пользователи.\n\n"
        f"Отправить <b>{len(user_stats)}</b> получателям?",
        reply_markup=b.as_markup(),
    )


async def _run_broadcast(msg: Message, bot: Bot, broadcast_text: str) -> None:
    uid = _uid(msg)
    admin_state.pop(uid, None)
    all_uids = list(user_stats.keys())
    total = len(all_uids)
    status_msg = await msg.answer(f"📢 Начинаю рассылку для <b>{total}</b> пользователей...")

    # Конвертируем Markdown → HTML (админ может писать **жирный**, *курсив*,
    # `код`, [ссылка](url), > цитата и т.д.). HTML-теги тоже работают насквозь.
    html_text = _md_to_html(broadcast_text)

    sent = failed = 0
    async with broadcast_lock:
        for target_uid in all_uids:
            try:
                await bot.send_message(target_uid, html_text, parse_mode=ParseMode.HTML)
                sent += 1
            except Exception:
                # Fallback: если HTML не парсится — шлём как plain text
                try:
                    await bot.send_message(target_uid, broadcast_text, parse_mode=None)
                    sent += 1
                except Exception:
                    failed += 1
            await asyncio.sleep(0.05)
    try:
        await status_msg.edit_text(
            f"✅ <b>Рассылка завершена</b>\n\n"
            f"📤 Отправлено: <b>{sent}</b>\n"
            f"❌ Ошибок: <b>{failed}</b>"
        )
    except Exception:
        pass


# ══════════════════════════════════════════════════════════════
# ОБРАБОТКА — ФОТО / ИЗОБРАЖЕНИЯ (всегда категория Vision)
# ══════════════════════════════════════════════════════════════

def _build_image_content(b64: str, mime_subtype: str, caption: str) -> list[dict]:
    return [
        {"type": "image_url", "image_url": {"url": f"data:image/{mime_subtype};base64,{b64}"}},
        {"type": "text", "text": caption},
    ]


@router.message(F.photo)
async def on_photo(msg: Message, bot: Bot) -> None:
    uid = _uid(msg)

    _track_user(uid, msg.from_user)
    if not await _check_limit(msg, uid):
        return


    photo = msg.photo[-1]
    file = await bot.get_file(photo.file_id)
    data = await bot.download_file(file.file_path)
    b64 = base64.b64encode(data.read()).decode()

    caption = msg.caption or "Что на этом изображении? Опиши подробно."
    content = _build_image_content(b64, "jpeg", caption)

    # category не указываем: ask_ai сам увидит, что content — изображение, и
    # направит его на vision-модели независимо от выбранной категории.
    await _run_ai_turn(msg, bot, content, thinking_label="👁 <i>Анализирую изображение...</i>")


def _extract_document_text(data: bytes, filename: str, mime: str) -> Optional[str]:
    """Достаёт текст из присланного документа. None — формат не поддержан.

    Работает без обязательных зависимостей: PDF/DOCX читаются, только если
    установлены pypdf / python-docx (иначе — понятный отказ выше по стеку).
    """
    ext = os.path.splitext(filename)[1].lower()

    # PDF
    if ext == ".pdf" or mime == "application/pdf":
        try:
            from pypdf import PdfReader
        except ImportError:
            return None
        try:
            reader = PdfReader(io.BytesIO(data))
            pages = [(p.extract_text() or "") for p in reader.pages]
            return "\n".join(pages).strip()
        except Exception as e:
            logging.warning(f"PDF extract error: {e}")
            return ""

    # DOCX
    if ext == ".docx" or mime == (
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    ):
        try:
            import docx  # python-docx
        except ImportError:
            return None
        try:
            document = docx.Document(io.BytesIO(data))
            return "\n".join(p.text for p in document.paragraphs).strip()
        except Exception as e:
            logging.warning(f"DOCX extract error: {e}")
            return ""

    # Всё текстовое (код, разметка, данные) + любой text/* mime
    if ext in TEXT_DOC_EXTS or mime.startswith("text/"):
        try:
            return data.decode("utf-8", errors="replace").strip()
        except Exception:
            return ""

    return None


@router.message(F.document)
async def on_document(msg: Message, bot: Bot) -> None:
    uid = _uid(msg)
    mime = msg.document.mime_type or ""
    filename = msg.document.file_name or "file"

    _track_user(uid, msg.from_user)
    if not await _check_limit(msg, uid):
        return


    file = await bot.get_file(msg.document.file_id)
    raw = await bot.download_file(file.file_path)
    data = raw.read()

    # 1) Картинка-документ — как и раньше, на vision-модели.
    if mime.startswith("image/"):
        b64 = base64.b64encode(data).decode()
        ext = mime.split("/")[-1]
        caption = msg.caption or "Опиши что на изображении."
        content = _build_image_content(b64, ext, caption)
        await _run_ai_turn(msg, bot, content, thinking_label="👁 <i>Анализирую...</i>")
        return

    # 2) Текстовый документ (txt/pdf/docx/код/данные) — извлекаем и отдаём модели.
    text = await asyncio.to_thread(_extract_document_text, data, filename, mime)

    if text is None:
        await msg.answer(
            "📄 Не поддерживаю такой формат. Пришлите текст, PDF, DOCX, "
            "изображение, аудио или голосовое сообщение."
        )
        return
    if not text.strip():
        await msg.answer(
            "📄 Не удалось извлечь текст из файла — возможно, это скан "
            "(изображение внутри PDF) или файл пустой."
        )
        return

    truncated = len(text) > MAX_DOC_CHARS
    if truncated:
        text = text[:MAX_DOC_CHARS]

    note = "\n\n[⚠️ Документ обрезан — показано начало.]" if truncated else ""
    caption = (msg.caption or "").strip() or "Разбери содержимое этого документа и кратко изложи суть."
    prompt = (
        f"{caption}\n\n"
        f"Содержимое файла «{filename}»:\n\n"
        f"{text}{note}"
    )

    await _run_ai_turn(msg, bot, prompt, thinking_label="📄 <i>Читаю документ...</i>")


# ══════════════════════════════════════════════════════════════
# ОБРАБОТКА — ГОЛОСОВЫЕ (STT → AI)
# ══════════════════════════════════════════════════════════════

async def _transcribe_with_indicator(bot: Bot, msg: Message, file_id: str, filename: str) -> str:
    async with TypingIndicator(bot, msg.chat.id):
        file = await bot.get_file(file_id)
        data = await bot.download_file(file.file_path)
        return await transcribe(data.read(), filename)


async def _handle_audio(msg: Message, bot: Bot, file_id: str, filename: str) -> None:
    uid = _uid(msg)

    _track_user(uid, msg.from_user)
    if not await _check_limit(msg, uid):
        return


    text = await _transcribe_with_indicator(bot, msg, file_id, filename)
    if not text:
        await msg.answer("❌ Не удалось распознать аудио.")
        return

    preview = text[:200] + ("..." if len(text) > 200 else "")
    await msg.answer(f"📝 <i>Распознано:</i> {html.quote(preview)}")

    await _run_ai_turn(msg, bot, text)


@router.message(F.voice)
async def on_voice(msg: Message, bot: Bot) -> None:
    await _handle_audio(msg, bot, msg.voice.file_id, "voice.ogg")


@router.message(F.video_note)
async def on_video_note(msg: Message, bot: Bot) -> None:
    await _handle_audio(msg, bot, msg.video_note.file_id, "video_note.mp4")


@router.message(F.audio)
async def on_audio(msg: Message, bot: Bot) -> None:
    fname = msg.audio.file_name or "audio.mp3"
    await _handle_audio(msg, bot, msg.audio.file_id, fname)


@router.message(F.video)
async def on_video(msg: Message, bot: Bot) -> None:
    uid = _uid(msg)

    _track_user(uid, msg.from_user)
    if not await _check_limit(msg, uid):
        return


    text = await _transcribe_with_indicator(bot, msg, msg.video.file_id, "video.mp4")
    if not text:
        await msg.answer("❌ Не удалось распознать аудио из видео.")
        return

    caption = msg.caption or ""
    full_text = f"{caption}\n\nТранскрипция видео:\n{text}" if caption else f"Транскрипция видео:\n{text}"

    preview = text[:200] + ("..." if len(text) > 200 else "")
    await msg.answer(f"📝 <i>Распознано:</i> {html.quote(preview)}")

    await _run_ai_turn(msg, bot, full_text)


# ══════════════════════════════════════════════════════════════
# ТРЕКИНГ ПОЛЬЗОВАТЕЛЕЙ
# ══════════════════════════════════════════════════════════════

def _track_user(uid: int, user) -> None:
    is_new = uid not in user_stats
    if is_new:
        user_stats[uid] = {
            "username": getattr(user, "username", None),
            "full_name": getattr(user, "full_name", str(uid)),
            "first_seen": datetime.now(timezone.utc),
            "total_requests": 0,
        }
        # Разовый стартовый баланс — строго один раз за всю жизнь аккаунта.
        # Условие на welcome_granted, а не на «нет в user_stats»: user_stats
        # удаляется для неактивных через 7 дней, и подарок выдавался снова.
        if uid not in ADMIN_IDS and uid not in welcome_granted:
            _get_usage(uid)["bought"] += WELCOME_TOKENS
            welcome_granted.add(uid)
        save_state()
    user_stats[uid]["total_requests"] += 1
    if user:
        user_stats[uid]["username"] = getattr(user, "username", None)
        user_stats[uid]["full_name"] = getattr(user, "full_name", str(uid))


# ══════════════════════════════════════════════════════════════
# АДМИН-ПАНЕЛЬ
# ══════════════════════════════════════════════════════════════

admin_state: dict[int, str] = {}
admin_pending_broadcast: dict[int, str] = {}  # uid -> текст рассылки до подтверждения


def _is_admin(uid: int) -> bool:
    return uid in ADMIN_IDS


def _admin_kb() -> InlineKeyboardBuilder:
    b = InlineKeyboardBuilder()
    b.row(
        InlineKeyboardButton(text="📊 Статистика", callback_data="admin:stats"),
        InlineKeyboardButton(text="💰 Продажи", callback_data="admin:sales"),
    )
    b.row(
        InlineKeyboardButton(text="👥 Пользователи", callback_data="admin:users"),
        InlineKeyboardButton(text="🔍 Найти юзера", callback_data="admin:find"),
    )
    b.row(
        InlineKeyboardButton(text="🎟 Промокоды", callback_data="admin:promo"),
        InlineKeyboardButton(text="🏷 Скидка", callback_data="admin:sale"),
    )
    b.row(
        InlineKeyboardButton(text="📢 Рассылка", callback_data="admin:broadcast"),
        InlineKeyboardButton(text="📥 Экспорт CSV", callback_data="admin:export"),
    )
    b.row(
        InlineKeyboardButton(text="📄 Код бота", callback_data="admin:file_code"),
        InlineKeyboardButton(text="💾 Бэкап БД", callback_data="admin:file_db"),
        InlineKeyboardButton(text="⚙️ .env", callback_data="admin:file_env"),
    )
    b.row(InlineKeyboardButton(text="📰 Тест сводки новостей", callback_data="admin:news_test"))
    b.row(
        InlineKeyboardButton(text="🛠 Изменить код (AI)", callback_data="admin:selfedit",
                             style="primary"),
        InlineKeyboardButton(text="♻️ Рестарт", callback_data="admin:restart",
                             style="danger"),
    )
    b.row(
        InlineKeyboardButton(text="⚙️ Лимиты", callback_data="admin:limits"),
        InlineKeyboardButton(text="🧩 Модели", callback_data="admin:models"),
    )
    b.row(InlineKeyboardButton(text="➕ Добавить ВСЕМ", callback_data="admin:give_all"))
    b.row(InlineKeyboardButton(text="🗑 Сбросить всю историю", callback_data="admin:clear_all",
                               style="danger"))
    closed_text = "🔴 ЗАКРЫТЬ ПРОЕКТ" if not project_closed else "🟢 ОТКРЫТЬ ПРОЕКТ"
    b.row(InlineKeyboardButton(text=closed_text, callback_data="admin:project_close",
                               style="danger"))
    return b


# ── АДМИН: КАРТОЧКА ПОЛЬЗОВАТЕЛЯ / ПОИСК / БАН / БАЛАНС ────────

def _find_uid(query: str) -> Optional[int]:
    """Ищет пользователя по ID или @username среди известных боту."""
    q = (query or "").strip()
    if q.startswith("@"):
        q_low = q[1:].lower()
        for uid, s in user_stats.items():
            if (s.get("username") or "").lower() == q_low:
                return uid
        return None
    if q.isdigit():
        uid = int(q)
        return uid if (uid in user_stats or uid in usage) else None
    return None


def _user_purchases(uid: int, limit: int = 5) -> list[dict]:
    return [p for p in reversed(purchases) if p["uid"] == uid][:limit]


def _admin_user_card_text(uid: int) -> str:
    s = user_stats.get(uid, {})
    uname = f"@{s['username']}" if s.get("username") else s.get("full_name", "—")
    u = usage.get(uid, {})
    used_today = u.get("used", 0) if u.get("date") == date.today() else 0
    plan = user_plan(uid)
    prem_line = (
        f"💎 Premium до: <b>{premium_until[uid].strftime('%d.%m.%Y')}</b>\n"
        if premium_active(uid) else ""
    )
    pays = _user_purchases(uid)
    if pays:
        pay_lines = "\n".join(
            f"  • {p['ts'][:10]} — {html.quote(str(p['title']))} — {p['amount']} "
            f"{'⭐' if p['currency'] == 'XTR' else '₽'}"
            for p in pays
        )
    else:
        pay_lines = "  —"
    return (
        f"👤 <b>{html.quote(str(uname))}</b> (<code>{uid}</code>)\n\n"
        f"💳 Тариф: <b>{plan.name}</b>\n{prem_line}"
        f"🚫 Бан: <b>{'да' if uid in banned_users else 'нет'}</b>\n"
        f"📊 Сегодня: <b>{used_today}</b> | всего: <b>{s.get('total_requests', 0)}</b>\n"
        f"💰 Баланс: <b>{fmt_tokens(u.get('bought', 0))}</b> токенов\n"
        f"👥 Приглашено друзей: <b>{referral_count.get(uid, 0)}</b>\n"
        f"🎫 Триал использован: <b>{'да' if uid in trial_used else 'нет'}</b>\n\n"
        f"🧾 <b>Последние покупки:</b>\n{pay_lines}"
    )


def _admin_user_kb(uid: int) -> InlineKeyboardBuilder:
    b = InlineKeyboardBuilder()
    b.row(
        InlineKeyboardButton(text="➕10k", callback_data=f"admu:{uid}:give:10000"),
        InlineKeyboardButton(text="➕50k", callback_data=f"admu:{uid}:give:50000"),
        InlineKeyboardButton(text="➕200k", callback_data=f"admu:{uid}:give:200000"),
    )
    b.row(
        InlineKeyboardButton(text="➖10k", callback_data=f"admu:{uid}:take:10000"),
        InlineKeyboardButton(text="➖50k", callback_data=f"admu:{uid}:take:50000"),
        InlineKeyboardButton(text="➖200k", callback_data=f"admu:{uid}:take:200000"),
    )
    b.row(
        InlineKeyboardButton(text=f"💎 +{PREMIUM_DAYS} дн.", callback_data=f"admu:{uid}:prem:{PREMIUM_DAYS}"),
        InlineKeyboardButton(text="💎 Забрать", callback_data=f"admu:{uid}:unprem:0"),
    )
    if uid in banned_users:
        b.row(InlineKeyboardButton(text="✅ Разбанить", callback_data=f"admu:{uid}:unban:0"))
    else:
        b.row(InlineKeyboardButton(text="🚫 Забанить", callback_data=f"admu:{uid}:ban:0"))
    b.row(InlineKeyboardButton(text="🔄 Сбросить дневной счётчик", callback_data=f"admu:{uid}:reset:0"))
    b.row(
        InlineKeyboardButton(text="💬 Написать", callback_data=f"admdm:{uid}"),
        InlineKeyboardButton(text="↩️ Возврат ⭐", callback_data=f"admrefund:{uid}"),
    )
    b.row(InlineKeyboardButton(text="⬅️ В админку", callback_data="admin:menu"))
    return b


async def _admin_show_user(msg: Message, uid: int, *, edit: bool = False) -> None:
    text = _admin_user_card_text(uid)
    kb = _admin_user_kb(uid).as_markup()
    if edit:
        try:
            await msg.edit_text(text, reply_markup=kb)
            return
        except Exception:
            pass
    await msg.answer(text, reply_markup=kb)


async def _admin_handle_search(msg: Message) -> None:
    uid = _find_uid(msg.text or "")
    if uid is None:
        await msg.answer("😕 Пользователь не найден. Нужен ID или @username того, кто уже писал боту.")
        return
    await _admin_show_user(msg, uid)


@router.callback_query(F.data.startswith("admu:"))
async def cb_admin_user_action(cb: CallbackQuery) -> None:
    if not _is_admin(cb.from_user.id):
        await cb.answer("⛔", show_alert=True)
        return
    try:
        _, uid_s, action, val_s = cb.data.split(":")
        uid, val = int(uid_s), int(val_s)
    except Exception:
        await cb.answer("Ошибка данных", show_alert=True)
        return
    note = "Готово"
    if action == "give":
        _grant_requests(uid, val)
        note = f"➕ Выдано {fmt_tokens(val)} ткн"
    elif action == "take":
        _take_requests(uid, val)
        note = f"➖ Списано {fmt_tokens(val)} ткн"
    elif action == "prem":
        until = grant_premium(uid, val)
        note = f"💎 Premium до {until.strftime('%d.%m.%Y')}"
    elif action == "unprem":
        premium_until.pop(uid, None)
        note = "💎 Premium снят"
    elif action == "ban":
        banned_users.add(uid)
        note = "🚫 Забанен"
    elif action == "unban":
        banned_users.discard(uid)
        note = "✅ Разбанен"
    elif action == "reset":
        _get_usage(uid)["used"] = 0
        note = "🔄 Дневной счётчик сброшен"
    save_state()
    await cb.answer(note)
    await _admin_show_user(cb.message, uid, edit=True)


# ── АДМИН: ВОЗВРАТ STARS-ПЛАТЕЖА ───────────────────────────────

def _user_star_charges(uid: int) -> list[tuple[str, dict]]:
    """Stars-платежи пользователя, новые первыми. Порядок стабильный —
    по нему строятся индексы в callback_data (сам charge_id туда не влезает)."""
    return sorted(
        ((cid, p) for cid, p in stars_charges.items() if p.get("uid") == uid),
        key=lambda x: x[1].get("ts", ""), reverse=True,
    )


def _charge_title(p: dict) -> str:
    payload = p.get("payload", "")
    if payload.startswith("premium"):
        return "Premium"
    if payload == "topup_stars":
        return "Пополнение"
    return "Пакет токенов"


@router.callback_query(F.data.startswith("admrefund:"))
async def cb_admin_refund_list(cb: CallbackQuery) -> None:
    """Список Stars-платежей пользователя с кнопками возврата."""
    if not _is_admin(cb.from_user.id):
        await cb.answer("⛔", show_alert=True)
        return
    uid = int(cb.data.split(":")[1])
    charges = _user_star_charges(uid)[:10]
    if not charges:
        await cb.answer("У пользователя нет Stars-платежей", show_alert=True)
        return
    await cb.answer()
    b = InlineKeyboardBuilder()
    lines = [f"↩️ <b>Возврат Stars</b> — <code>{uid}</code>\n"]
    for i, (cid, p) in enumerate(charges):
        mark = "✅ возвращён" if p.get("refunded") else "⭐ можно вернуть"
        lines.append(
            f"{i + 1}. {p.get('ts', '')[:10]} · {_charge_title(p)} · "
            f"{p.get('stars', 0)} ⭐ — {mark}")
        if not p.get("refunded"):
            b.row(InlineKeyboardButton(
                text=f"↩️ Вернуть №{i + 1} ({p.get('stars', 0)} ⭐)",
                callback_data=f"admrf:{uid}:{i}"))
    b.row(InlineKeyboardButton(text="⬅️ К карточке", callback_data=f"admu:{uid}:reset_view:0"))
    lines.append("\nВозврат уходит сразу и отмене не подлежит. "
                 "Начисленные токены/Premium при возврате снимаются.")
    await cb.message.answer("\n".join(lines), reply_markup=b.as_markup())


@router.callback_query(F.data.startswith("admrf:"))
async def cb_admin_refund_do(cb: CallbackQuery) -> None:
    """Выполняет возврат: refundStarPayment + снятие начисленного."""
    if not _is_admin(cb.from_user.id):
        await cb.answer("⛔", show_alert=True)
        return
    _, uid_s, idx_s = cb.data.split(":")
    uid, idx = int(uid_s), int(idx_s)
    charges = _user_star_charges(uid)
    if idx >= len(charges):
        await cb.answer("Платёж не найден", show_alert=True)
        return
    charge_id, p = charges[idx]
    if p.get("refunded"):
        await cb.answer("Уже возвращён", show_alert=True)
        return
    try:
        await cb.message.bot.refund_star_payment(
            user_id=uid, telegram_payment_charge_id=charge_id)
    except Exception as e:
        await cb.answer(f"Telegram отказал: {str(e)[:150]}", show_alert=True)
        return

    p["refunded"] = True
    # Снимаем то, что было начислено этим платежом.
    payload = p.get("payload", "")
    undone = ""
    try:
        if payload.startswith("premium"):
            premium_until.pop(uid, None)
            undone = "Premium снят"
        elif payload == "topup_stars":
            tokens = int(p.get("stars", 0)) * TOKENS_PER_STAR
            _take_requests(uid, tokens)
            undone = f"списано {fmt_tokens(tokens)} ткн"
        else:
            pack = _pack_from_payload(payload)
            if pack is None:
                undone = "пакет не распознан — начисление не списано"
            else:
                _take_requests(uid, pack.tokens)
                undone = f"списано {fmt_tokens(pack.tokens)} ткн"
    except Exception as e:
        undone = f"начисленное снять не удалось: {e}"
    save_state_now()  # денежная операция — пишем на диск сразу

    await cb.answer("✅ Возврат отправлен")
    await cb.message.answer(
        f"✅ Возврат <b>{p.get('stars', 0)} ⭐</b> пользователю <code>{uid}</code> "
        f"выполнен ({undone}).")
    try:
        await cb.message.bot.send_message(
            uid,
            f"↩️ Вам возвращён платёж на <b>{p.get('stars', 0)} ⭐</b>. "
            "Звёзды вернутся на ваш баланс Telegram.")
    except Exception:
        pass


# ── АДМИН: НАПИСАТЬ ПОЛЬЗОВАТЕЛЮ ───────────────────────────────

@router.callback_query(F.data.startswith("admdm:"))
async def cb_admin_dm(cb: CallbackQuery) -> None:
    if not _is_admin(cb.from_user.id):
        await cb.answer("⛔", show_alert=True)
        return
    uid = int(cb.data.split(":")[1])
    admin_state[cb.from_user.id] = f"waiting_dm:{uid}"
    await cb.answer()
    b = InlineKeyboardBuilder()
    b.row(InlineKeyboardButton(text="❌ Отмена", callback_data="admin:broadcast_cancel"))
    await cb.message.answer(
        f"💬 <b>Сообщение пользователю</b> <code>{uid}</code>\n\n"
        "Пришлите текст — я отправлю его от имени бота.",
        reply_markup=b.as_markup(),
    )


async def _admin_handle_dm(msg: Message, target_uid: int) -> None:
    text = (msg.text or "").strip()
    if not text:
        await msg.answer("😕 Пустое сообщение, отправка отменена.")
        return
    try:
        await msg.bot.send_message(
            target_uid,
            f"✉️ <b>Сообщение от администратора:</b>\n\n{html.quote(text)}")
        await msg.answer(f"✅ Отправлено пользователю <code>{target_uid}</code>.")
    except Exception as e:
        await msg.answer(
            f"😕 Не доставлено (пользователь мог заблокировать бота): "
            f"<code>{html.quote(str(e)[:200])}</code>")


@router.callback_query(F.data == "admin:find")
async def cb_admin_find(cb: CallbackQuery) -> None:
    if not _is_admin(cb.from_user.id):
        await cb.answer("⛔", show_alert=True)
        return
    await cb.answer()
    admin_state[cb.from_user.id] = "waiting_user_search"
    b = InlineKeyboardBuilder()
    b.row(InlineKeyboardButton(text="❌ Отмена", callback_data="admin:broadcast_cancel"))
    try:
        await cb.message.edit_text(
            "🔍 <b>Поиск пользователя</b>\n\nПришлите ID или @username.",
            reply_markup=b.as_markup(),
        )
    except Exception:
        pass


def _parse_admin_target(msg: Message, need_amount: bool = True) -> Optional[tuple]:
    """Разбирает '/cmd <uid|@username> [число]'. Возвращает (uid, число) или None."""
    parts = (msg.text or "").split()
    if len(parts) < 2:
        return None
    uid = _find_uid(parts[1])
    if uid is None:
        return None
    amount = 0
    if len(parts) > 2 and parts[2].isdigit():
        amount = int(parts[2])
    elif need_amount:
        return None
    return uid, amount


@router.message(Command("user"))
async def cmd_admin_user(msg: Message) -> None:
    if not _is_admin(_uid(msg)):
        return
    res = _parse_admin_target(msg, need_amount=False)
    if not res:
        await msg.answer("Формат: <code>/user ID_или_@username</code>")
        return
    await _admin_show_user(msg, res[0])


@router.message(Command("give"))
async def cmd_give(msg: Message) -> None:
    if not _is_admin(_uid(msg)):
        return
    res = _parse_admin_target(msg)
    if not res:
        await msg.answer("Формат: <code>/give ID_или_@username ЧИСЛО_ТОКЕНОВ</code>")
        return
    uid, amount = res
    _grant_requests(uid, amount)
    save_state()
    await msg.answer(f"✅ Пользователю <code>{uid}</code> выдано <b>{fmt_tokens(amount)}</b> токенов.")


@router.message(Command("take"))
async def cmd_take(msg: Message) -> None:
    if not _is_admin(_uid(msg)):
        return
    res = _parse_admin_target(msg)
    if not res:
        await msg.answer("Формат: <code>/take ID_или_@username ЧИСЛО_ТОКЕНОВ</code>")
        return
    uid, amount = res
    _take_requests(uid, amount)
    save_state()
    await msg.answer(f"✅ У пользователя <code>{uid}</code> списано <b>{fmt_tokens(amount)}</b> токенов.")


@router.message(Command("prem"))
async def cmd_prem(msg: Message) -> None:
    if not _is_admin(_uid(msg)):
        return
    res = _parse_admin_target(msg, need_amount=False)
    if not res:
        await msg.answer("Формат: <code>/prem ID_или_@username [ДНЕЙ]</code>")
        return
    uid, days = res
    until = grant_premium(uid, days or PREMIUM_DAYS)
    save_state()
    await msg.answer(f"💎 Premium пользователю <code>{uid}</code> до <b>{until.strftime('%d.%m.%Y')}</b>.")


@router.message(Command("ban"))
async def cmd_ban(msg: Message) -> None:
    if not _is_admin(_uid(msg)):
        return
    res = _parse_admin_target(msg, need_amount=False)
    if not res:
        await msg.answer("Формат: <code>/ban ID_или_@username</code>")
        return
    banned_users.add(res[0])
    save_state()
    await msg.answer(f"🚫 Пользователь <code>{res[0]}</code> забанен.")


@router.message(Command("unban"))
async def cmd_unban(msg: Message) -> None:
    if not _is_admin(_uid(msg)):
        return
    parts = (msg.text or "").split()
    if len(parts) < 2:
        await msg.answer("Формат: <code>/unban ID_или_@username</code>")
        return
    uid = _find_uid(parts[1])
    if uid is None and parts[1].isdigit():
        uid = int(parts[1])
    if uid is None:
        await msg.answer("Пользователь не найден.")
        return
    banned_users.discard(uid)
    banned_notified.discard(uid)  # при новом бане снова покажем уведомление
    save_state()
    await msg.answer(f"✅ Пользователь <code>{uid}</code> разбанен.")


# ── АДМИН: ПРОДАЖИ ──────────────────────────────────────

def _sales_sum(records: list) -> str:
    stars = sum(p["amount"] for p in records if p["currency"] == "XTR")
    rub = sum(p["amount"] for p in records if p["currency"] == "RUB")
    return f"{len(records)} шт. — {stars} ⭐ + {rub} ₽"


def _admin_sales_text() -> str:
    now = datetime.now(timezone.utc)

    def since(days: int) -> list:
        edge = now - timedelta(days=days)
        out = []
        for p in purchases:
            try:
                if datetime.fromisoformat(p["ts"]) >= edge:
                    out.append(p)
            except Exception:
                continue
        return out

    last = list(reversed(purchases))[:10]
    last_lines = "\n".join(
        f"• {p['ts'][:16].replace('T', ' ')} — <code>{p['uid']}</code> — "
        f"{html.quote(str(p['title']))} — {p['amount']} {'⭐' if p['currency'] == 'XTR' else '₽'}"
        for p in last
    ) or "—"
    return (
        "💰 <b>Продажи</b>\n\n"
        f"📅 За 24 часа: <b>{_sales_sum(since(1))}</b>\n"
        f"🗓 7 дней: <b>{_sales_sum(since(7))}</b>\n"
        f"🗓 30 дней: <b>{_sales_sum(since(30))}</b>\n"
        f"📦 За всё время: <b>{_sales_sum(purchases)}</b>\n\n"
        f"🧾 <b>Последние покупки:</b>\n{last_lines}"
    )


@router.callback_query(F.data == "admin:sales")
async def cb_admin_sales(cb: CallbackQuery) -> None:
    if not _is_admin(cb.from_user.id):
        await cb.answer("⛔", show_alert=True)
        return
    await cb.answer()
    b = InlineKeyboardBuilder()
    b.row(InlineKeyboardButton(text="🔄 Обновить", callback_data="admin:sales"))
    b.row(InlineKeyboardButton(text="⬅️ Назад", callback_data="admin:menu"))
    try:
        await cb.message.edit_text(_admin_sales_text(), reply_markup=b.as_markup())
    except Exception:
        pass


# ── АДМИН: ПРОМОКОДЫ ────────────────────────────────────

def _admin_promo_text() -> str:
    if not promo_codes:
        codes = "—"
    else:
        lines = []
        for code, info in promo_codes.items():
            status = {"active": "🟢", "expired": "⚪ истёк", "exhausted": "⚪ исчерпан"}[_promo_status(info)]
            kind = "дн. Premium 💎" if info.get("kind") == "premium" else "токенов ⭐"
            max_uses = info.get("max_uses", 0)
            uses = f"{info.get('used', 0)}/{max_uses if max_uses else '∞'}"
            exp = f", до {info['expires']}" if info.get("expires") else ""
            lines.append(f"{status} <code>{code}</code> — {info.get('value', 0)} {kind} ({uses}{exp})")
        codes = "\n".join(lines)
    return (
        "🎟 <b>Промокоды</b>\n\n" + codes +
        "\n\nУдалить: <code>/delpromo КОД</code>"
    )


@router.callback_query(F.data == "admin:promo")
async def cb_admin_promo(cb: CallbackQuery) -> None:
    if not _is_admin(cb.from_user.id):
        await cb.answer("⛔", show_alert=True)
        return
    await cb.answer()
    b = InlineKeyboardBuilder()
    b.row(InlineKeyboardButton(text="➕ Создать промокод", callback_data="admin:promo_new"))
    b.row(InlineKeyboardButton(text="⬅️ Назад", callback_data="admin:menu"))
    try:
        await cb.message.edit_text(_admin_promo_text(), reply_markup=b.as_markup())
    except Exception:
        pass


@router.callback_query(F.data == "admin:promo_new")
async def cb_admin_promo_new(cb: CallbackQuery) -> None:
    if not _is_admin(cb.from_user.id):
        await cb.answer("⛔", show_alert=True)
        return
    await cb.answer()
    admin_state[cb.from_user.id] = "waiting_promo"
    b = InlineKeyboardBuilder()
    b.row(InlineKeyboardButton(text="❌ Отмена", callback_data="admin:broadcast_cancel"))
    try:
        await cb.message.edit_text(
            "➕ <b>Новый промокод</b>\n\nПришлите строку в формате:\n"
            "<code>КОД тип значение [макс_активаций] [дней_действия]</code>\n\n"
            "тип: <code>req</code> (токены) или <code>prem</code> (дни Premium)\n\n"
            "Примеры:\n"
            "<code>SUMMER req 50000 100 7</code> — 50 000 токенов, 100 активаций, 7 дней\n"
            "<code>VIP prem 30 10</code> — 30 дн. Premium, 10 активаций, бессрочно",
            reply_markup=b.as_markup(),
        )
    except Exception:
        pass


async def _admin_handle_promo_create(msg: Message) -> None:
    parts = (msg.text or "").split()
    if len(parts) < 3 or parts[1].lower() not in ("req", "prem") or not parts[2].isdigit():
        await msg.answer("😕 Неверный формат. Пример: <code>SUMMER req 50000 100 7</code>")
        return
    code = parts[0].upper()
    kind = "premium" if parts[1].lower() == "prem" else "requests"
    value = int(parts[2])
    max_uses = int(parts[3]) if len(parts) > 3 and parts[3].isdigit() else 0
    expires = None
    if len(parts) > 4 and parts[4].isdigit():
        expires = (date.today() + timedelta(days=int(parts[4]))).isoformat()
    promo_codes[code] = {"kind": kind, "value": value, "max_uses": max_uses,
                         "used": 0, "expires": expires, "users": []}
    save_state()
    await msg.answer(
        f"✅ Промокод <code>{code}</code> создан!\n"
        f"Пользователи активируют его командой:\n<code>/promo {code}</code>"
    )


@router.message(Command("delpromo"))
async def cmd_delpromo(msg: Message) -> None:
    if not _is_admin(_uid(msg)):
        return
    parts = (msg.text or "").split()
    if len(parts) < 2:
        await msg.answer("Формат: <code>/delpromo КОД</code>")
        return
    code = parts[1].upper()
    if promo_codes.pop(code, None) is None:
        await msg.answer("Такого промокода нет.")
        return
    save_state()
    await msg.answer(f"🗑 Промокод <code>{code}</code> удалён.")


# ── АДМИН: СКИДКА / АКЦИЯ ────────────────────────────────

@router.callback_query(F.data == "admin:sale")
async def cb_admin_sale(cb: CallbackQuery) -> None:
    if not _is_admin(cb.from_user.id):
        await cb.answer("⛔", show_alert=True)
        return
    await cb.answer()
    admin_state[cb.from_user.id] = "waiting_sale"
    p = sale_percent()
    cur = f"Сейчас: <b>−{p}%</b> до {sale_info.get('until') or '∞'}" if p else "Сейчас скидки нет."
    b = InlineKeyboardBuilder()
    if p:
        b.row(InlineKeyboardButton(text="🛑 Отключить скидку", callback_data="admin:sale_off"))
    b.row(InlineKeyboardButton(text="❌ Отмена", callback_data="admin:broadcast_cancel"))
    try:
        await cb.message.edit_text(
            "🏷 <b>Скидка на магазин</b>\n\n" + cur + "\n\n"
            "Пришлите: <code>ПРОЦЕНТ [ДНЕЙ]</code>\n"
            "Например: <code>30 7</code> — скидка 30% на 7 дней\n"
            "или <code>20</code> — 20% бессрочно.",
            reply_markup=b.as_markup(),
        )
    except Exception:
        pass


@router.callback_query(F.data == "admin:sale_off")
async def cb_admin_sale_off(cb: CallbackQuery) -> None:
    if not _is_admin(cb.from_user.id):
        await cb.answer("⛔", show_alert=True)
        return
    sale_info.clear()
    admin_state.pop(cb.from_user.id, None)
    save_state()
    await cb.answer("Скидка отключена ✅", show_alert=True)


async def _admin_handle_sale(msg: Message) -> None:
    parts = (msg.text or "").split()
    if not parts or not parts[0].isdigit() or not (0 < int(parts[0]) < 100):
        await msg.answer("😕 Неверный формат. Пример: <code>30 7</code>")
        return
    percent = int(parts[0])
    until = None
    if len(parts) > 1 and parts[1].isdigit():
        until = (date.today() + timedelta(days=int(parts[1]))).isoformat()
    sale_info.clear()
    sale_info.update({"percent": percent, "until": until})
    save_state()
    till = f" до {until}" if until else ""
    await msg.answer(f"🏷 Скидка <b>−{percent}%</b>{till} включена! Цены в магазине обновлены.")


# ── АДМИН: ЭКСПОРТ CSV ──────────────────────────────────

@router.callback_query(F.data == "admin:export")
async def cb_admin_export(cb: CallbackQuery) -> None:
    if not _is_admin(cb.from_user.id):
        await cb.answer("⛔", show_alert=True)
        return
    await cb.answer("Готовлю файлы…")
    import csv
    import io
    from aiogram.types import BufferedInputFile

    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["uid", "username", "full_name", "total_requests", "used_today",
                "bought", "premium_until", "referrals", "banned", "first_seen"])
    today = date.today()
    for uid, s in user_stats.items():
        u = usage.get(uid, {})
        w.writerow([
            uid, s.get("username", ""), s.get("full_name", ""),
            s.get("total_requests", 0),
            u.get("used", 0) if u.get("date") == today else 0,
            u.get("bought", 0),
            premium_until[uid].isoformat() if uid in premium_until else "",
            referral_count.get(uid, 0),
            1 if uid in banned_users else 0,
            s["first_seen"].isoformat() if s.get("first_seen") else "",
        ])
    users_file = BufferedInputFile(buf.getvalue().encode("utf-8-sig"), filename="users.csv")

    buf2 = io.StringIO()
    w2 = csv.writer(buf2)
    w2.writerow(["ts", "uid", "kind", "title", "amount", "currency"])
    for p in purchases:
        w2.writerow([p["ts"], p["uid"], p["kind"], p["title"], p["amount"], p["currency"]])
    purchases_file = BufferedInputFile(buf2.getvalue().encode("utf-8-sig"), filename="purchases.csv")

    try:
        await cb.message.answer_document(users_file, caption="👥 Пользователи")
        await cb.message.answer_document(purchases_file, caption="💰 Покупки")
    except Exception as e:
        await cb.message.answer(f"😕 Не удалось отправить экспорт: {e}")


# ── АДМИН: ФАЙЛЫ С СЕРВЕРА (код бота / бэкап БД / любой файл) ──

TG_FILE_LIMIT = 50 * 1024 * 1024  # лимит Bot API на отправку документа — 50 МБ


def _within_bot_dirs(raw: str) -> Optional[str]:
    """Приводит путь к абсолютному и проверяет, что он внутри папки бота или
    рабочей папки. Возвращает None, если путь ведёт наружу.

    Обе папки разрешены потому, что бота запускают и из его каталога, и из
    другого места (тогда .env и БД лежат в рабочей папке). Сравнение через
    commonpath, а не startswith: '/botdir_evil' не должен пройти как '/botdir'.
    """
    roots = {os.path.abspath(os.path.dirname(os.path.abspath(__file__))), os.path.abspath(os.getcwd())}
    for root in roots:
        try:
            target = os.path.abspath(raw) if os.path.isabs(raw) else os.path.abspath(os.path.join(root, raw))
            # На Windows commonpath бросает ValueError для разных дисков (C: и D:).
            if os.path.commonpath([root, target]) == root:
                return target
        except ValueError:
            continue
    return None


async def _admin_send_file(msg: Message, path: str, caption: str = "",
                           uid: Optional[int] = None) -> None:
    """Отправляет файл с сервера в чат. Понятные ошибки вместо молчаливого падения.

    Через эту функцию уходят исходник, БД, .env и произвольный путь из /getfile,
    поэтому проверка прав продублирована здесь: цена одного забытого гварда у
    вызывающего — выдача любого файла с сервера. uid передаётся явно, потому что
    в callback-хендлерах msg — это cb.message, где from_user это сам бот.
    """
    if uid is not None and not _is_admin(uid):
        return
    if not os.path.isfile(path):
        await msg.answer(f"😕 Файл не найден: <code>{html.quote(path)}</code>")
        return
    size = os.path.getsize(path)
    if size > TG_FILE_LIMIT:
        await msg.answer(
            f"😕 Файл слишком большой для Telegram: {size / 1024 / 1024:.1f} МБ (лимит 50 МБ)."
        )
        return
    try:
        await msg.answer_document(
            FSInputFile(path),
            caption=caption or f"📄 <code>{html.quote(os.path.abspath(path))}</code>",
        )
    except Exception as e:
        await msg.answer(f"😕 Не удалось отправить файл: {html.quote(str(e))}")


@router.callback_query(F.data == "admin:file_code")
async def cb_admin_file_code(cb: CallbackQuery) -> None:
    """Присылает текущий исходник бота (тот файл, что запущен)."""
    if not _is_admin(cb.from_user.id):
        await cb.answer("⛔", show_alert=True)
        return
    await cb.answer("Отправляю код…")
    await _admin_send_file(cb.message, os.path.abspath(__file__),
                           caption="📄 Исходный код бота (запущенная версия)",
                           uid=cb.from_user.id)


@router.callback_query(F.data == "admin:file_db")
async def cb_admin_file_db(cb: CallbackQuery) -> None:
    """Присылает свежий бэкап SQLite-базы (перед отправкой сбрасывает состояние на диск)."""
    if not _is_admin(cb.from_user.id):
        await cb.answer("⛔", show_alert=True)
        return
    await cb.answer("Делаю бэкап…")
    save_state_now()  # чтобы в файле были самые свежие данные, а не минутной давности
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    await _admin_send_file(cb.message, DB_FILE, caption=f"💾 Бэкап базы на {stamp}",
                           uid=cb.from_user.id)


@router.callback_query(F.data == "admin:file_env")
async def cb_admin_file_env(cb: CallbackQuery) -> None:
    """Первый шаг выгрузки .env: предупреждение и подтверждение.

    Двухшаговым сделано 22.08.2026: кнопка стоит в общей клавиатуре рядом с
    безобидными разделами, а один промах пальцем публикует ВСЕ токены в облако
    Telegram навсегда (file_id живёт и после удаления сообщения). Сброс историй
    подтверждение имеет — выгрузка секретов тем более обязана.
    """
    if not _is_admin(cb.from_user.id):
        await cb.answer("⛔", show_alert=True)
        return
    b = InlineKeyboardBuilder()
    b.row(InlineKeyboardButton(text="⚠️ Да, прислать секреты", callback_data="admin:file_env_go"))
    b.row(InlineKeyboardButton(text="❌ Отмена", callback_data="admin:menu"))
    await cb.answer()
    try:
        await cb.message.edit_text(
            "⚠️ <b>Выгрузка .env</b>\n━━━━━━━━━━━━━━━\n"
            "В файле <b>все токены бота</b>: BOT_TOKEN, ключи провайдеров, "
            "PLATEGA_SECRET, PANEL_API_KEY.\n\n"
            "Загруженный в Telegram файл остаётся в облаке <b>навсегда</b> и "
            "доступен по file_id, даже если удалить сообщение.\n\n"
            "<i>Продолжить?</i>",
            reply_markup=b.as_markup(),
        )
    except Exception as e:
        logging.debug(f"file_env: не удалось показать подтверждение: {e}")


@router.callback_query(F.data == "admin:file_env_go")
async def cb_admin_file_env_go(cb: CallbackQuery) -> None:
    """Второй шаг: собственно отправка .env после подтверждения."""
    if not _is_admin(cb.from_user.id):
        await cb.answer("⛔", show_alert=True)
        return
    await cb.answer("Отправляю .env…")
    # .env лежит рядом с ботом; если запуск шёл из другой папки — ищем и там.
    path = ".env"
    if not os.path.isfile(path):
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    await _admin_send_file(
        cb.message, path,
        caption="⚙️ .env — здесь ВСЕ секреты бота. Не пересылайте никому.",
        uid=cb.from_user.id,
    )


@router.callback_query(F.data == "admin:news_test")
async def cb_admin_news_test(cb: CallbackQuery) -> None:
    """Прямо сейчас собирает и присылает новостную сводку (тест ежедневной рассылки)."""
    if not _is_admin(cb.from_user.id):
        await cb.answer("⛔", show_alert=True)
        return
    await cb.answer("Собираю новости, это займёт до минуты…")
    try:
        await _send_news_digest(cb.message.bot)
    except Exception as e:
        logging.warning("Тест сводки новостей: %s", e)
        await cb.message.answer(
            f"😕 Сводка не собралась: <code>{html.quote(str(e)[:300])}</code>")


@router.message(Command("getfile"))
async def cmd_getfile(msg: Message) -> None:
    """/getfile <путь> — прислать файл бота (только админ).

    Примеры: /getfile bot_state.db · /getfile logs/bot.log
    Путь считается от папки бота или от рабочей папки; выход за их пределы
    запрещён (см. _within_bot_dirs).
    """
    uid = _uid(msg)
    if not _is_admin(uid):
        return  # для остальных команда «не существует»
    parts = (msg.text or "").split(maxsplit=1)
    if len(parts) < 2 or not parts[1].strip():
        await msg.answer(
            "📄 <b>Файл с сервера</b>\n\n"
            "Использование: <code>/getfile путь/к/файлу</code>\n"
            "Например: <code>/getfile bot_state.db</code>"
        )
        return
    raw = parts[1].strip().strip('"')
    target = _within_bot_dirs(raw)
    if target is None:
        # Не «защита от админа», а страховка от опечатки и от того, что команда
        # однажды окажется доступна не тому: /getfile ../../.ssh/id_rsa читал
        # что угодно на сервере. Мешает — убери проверку, она тут одна.
        await msg.answer(
            "😕 Путь вне папки бота — не отдаю.\n"
            "<i>Можно только файлы бота: bot_state.db, logs/bot.log и рядом.</i>"
        )
        return
    await _admin_send_file(msg, target, uid=uid)


# ══════════════════════════════════════════════════════════════
# АДМИН: САМО-РЕДАКТИРОВАНИЕ КОДА (Claude переписывает исходник)
# ══════════════════════════════════════════════════════════════
#
# /selfedit <инструкция> — админ описывает, что изменить. Бот читает свой
# исходник, отправляет его модели SELFEDIT_MODEL через EchoGate, получает
# ПОЛНЫЙ новый файл, проверяет синтаксис (compile) и сохраняет ЧЕРНОВИК в памяти. Запись
# на диск, бэкап и рестарт происходят отдельным шагом — после кнопки
# «Применить» (можно посмотреть diff или отменить, ничего не меняя).
# Панель play2go (Pterodactyl) поднимает процесс заново уже с новым кодом.
#
# Защита от «кирпича»: перед рестартом создаётся маркер .selfupdate_pending с
# путём к бэкапу и счётчиком попыток. Если новый код падает при старте и бот
# несколько раз подряд не выходит на связь — код автоматически откатывается к
# бэкапу. /rollback — ручной откат к последнему бэкапу.
#
# ⚠️ Ограничение: compile() ловит синтаксис, но НЕ ошибки уровня импорта,
# которые проявятся только при запуске нового кода (тогда сработает авто-откат
# по счётчику попыток). Правки затрагивают весь файл, поэтому только для админа.

SELFUPDATE_MARKER = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), ".selfupdate_pending"
)
SELFUPDATE_MAX_ATTEMPTS = 2      # неудачных стартов подряд до авто-отката
SELFUPDATE_HEALTH_DELAY = 25.0   # сек стабильной работы = обновление удачно

# Черновики self-edit: код от Claude сначала попадает сюда, а на сервер
# записывается только после явного подтверждения админом (двухэтапный режим).
# Хранится в памяти — при рестарте бота неприменённый черновик пропадает.
selfedit_drafts: dict[int, dict] = {}   # uid -> {code, source, instruction, ts}


def _selfupdate_system_prompt() -> str:
    return (
        "Ты — инженер, редактирующий исходный код Telegram-бота на Python "
        "(aiogram 3). Тебе дают ПОЛНЫЙ текущий файл бота и инструкцию, что "
        "изменить. Верни ПОЛНЫЙ обновлённый файл целиком, без сокращений, без "
        "«...» и без пояснений до или после. Сохрани всю существующую "
        "функциональность, стиль, комментарии и структуру — меняй только то, "
        "что просят. Код обязан быть синтаксически корректным и запускаться. "
        "Оберни весь файл в один блок ```python ... ```."
    )


def _extract_code(text: str) -> str:
    """Достаёт код из ответа Claude: содержимое ```python ...``` либо весь текст."""
    m = re.search(r"```(?:python)?\s*\n(.*?)```", text, re.DOTALL)
    code = (m.group(1) if m else text).strip()
    return code + "\n"


async def _claude_rewrite_source(source: str, instruction: str) -> str:
    """Отправляет исходник + инструкцию модели SELFEDIT_MODEL через EchoGate
    (OpenAI-совместимый стриминг), возвращает новый код. Стриминг обязателен —
    файл большой, ответ длинный, иначе таймаут. strip_think=False: в исходнике
    бота буквально встречается <think>...</think> (regex в _call_model), и
    фильтр размышлений повредил бы код. timeout увеличен: полная перегенерация
    файла с рассуждениями может идти несколько минут."""
    user_msg = (
        f"Инструкция: {instruction}\n\n"
        "Ниже — полный текущий файл бота. Верни его целиком с внесёнными "
        "изменениями.\n\n"
        f"```python\n{source}\n```"
    )
    text = await _call_model(
        SELFEDIT_MODEL,
        [
            {"role": "system", "content": _selfupdate_system_prompt()},
            {"role": "user", "content": user_msg},
        ],
        temperature=0.2,
        max_tokens=128000,
        provider="echogate",
        reasoning_effort="high",
        timeout=600,
        strip_think=False,
    )
    if not text.strip():
        raise RuntimeError("Пустой ответ от модели.")
    return _extract_code(text)


@router.message(Command("selfedit"))
async def cmd_selfedit(msg: Message) -> None:
    """/selfedit <инструкция> — Claude переписывает исходник бота (только админ)."""
    uid = _uid(msg)
    if not _is_admin(uid):
        return  # для остальных команда «не существует»
    if not ECHOGATE_KEY:
        await msg.answer(
            "⛔ Не задан <code>ECHOGATE_KEY</code> в .env — "
            "само-редактирование недоступно."
        )
        return
    parts = (msg.text or "").split(maxsplit=1)
    if len(parts) < 2 or not parts[1].strip():
        await msg.answer(
            "🛠 <b>Само-редактирование кода</b>\n\n"
            "Опишите, что изменить в боте:\n"
            "<code>/selfedit добавь команду /ping, отвечающую «pong»</code>\n\n"
            "Бот перепишет свой исходник через Claude и покажет черновик — "
            "запись на сервер и рестарт произойдут только после кнопки "
            "«Применить». При неудачном старте — авто-откат.\n"
            f"🧠 Модель: <code>{html.quote(SELFEDIT_MODEL)}</code> (EchoGate)\n"
            "↩️ Ручной откат: <code>/rollback</code>"
        )
        return

    instruction = parts[1].strip()
    await _do_selfedit(msg, instruction, uid)


async def _do_selfedit(msg: Message, instruction: str, uid: int) -> None:
    """Общая логика само-редактирования: Claude → проверки → бэкап → запись →
    рестарт. Используется и командой /selfedit, и кнопкой в админ-панели."""
    path = os.path.abspath(__file__)
    status = await msg.answer(
        "🧠 Claude читает код и вносит изменения… это может занять пару минут."
    )

    # 1) запрос к Claude
    try:
        with open(path, encoding="utf-8") as f:
            source = f.read()
        new_code = await _claude_rewrite_source(source, instruction)
    except Exception as e:
        await status.edit_text(f"😕 Ошибка обращения к Claude: {html.quote(str(e))}")
        return

    # 2) проверка синтаксиса
    try:
        compile(new_code, path, "exec")
    except SyntaxError as e:
        await status.edit_text(
            "⛔ Новый код не прошёл проверку синтаксиса — изменения отклонены.\n"
            f"<code>{html.quote(str(e))}</code>"
        )
        return

    # 3) защита от обрезанного ответа и потери точки входа
    if len(new_code) < len(source) * 0.5:
        await status.edit_text(
            "⛔ Новый файл подозрительно короткий (Claude мог обрезать код) — "
            "изменения отклонены ради безопасности."
        )
        return
    if 'if __name__ == "__main__":' not in new_code or "async def main(" not in new_code:
        await status.edit_text(
            "⛔ В новом коде не найдена точка входа (main / __main__) — "
            "изменения отклонены ради безопасности."
        )
        return

    # 4) сохраняем ЧЕРНОВИК — на диск ничего не пишем, ждём подтверждения админа
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    selfedit_drafts[uid] = {
        "code": new_code, "source": source, "instruction": instruction, "ts": ts,
    }
    added, removed = _selfedit_diff_stats(source, new_code)
    kb = InlineKeyboardBuilder()
    kb.row(InlineKeyboardButton(
        text="🚀 Применить и перезапустить", callback_data="admin:selfedit_apply",
        style="success"))
    kb.row(
        InlineKeyboardButton(text="📄 Diff", callback_data="admin:selfedit_diff",
                             style="primary"),
        InlineKeyboardButton(text="❌ Отменить", callback_data="admin:selfedit_discard",
                             style="danger"),
    )
    old_size = f"{len(source):,}".replace(",", " ")
    new_size = f"{len(new_code):,}".replace(",", " ")
    await status.edit_text(
        "📝 <b>Черновик готов</b>\n"
        "━━━━━━━━━━━━━━━\n"
        f"🛠 <i>{html.quote(instruction[:300])}</i>\n\n"
        f"➕ <b>{added}</b> строк · ➖ <b>{removed}</b> строк\n"
        f"📏 {old_size} → {new_size} символов\n"
        "━━━━━━━━━━━━━━━\n"
        "Код <b>ещё не применён</b>. «Применить» — бэкап, запись "
        "на сервер и рестарт; «Отменить» — всё останется как есть.",
        reply_markup=kb.as_markup(),
    )
    logging.info("🛠 selfedit: черновик готов (админ %s): %s", uid, instruction)


def _selfedit_diff_stats(old: str, new: str) -> tuple[int, int]:
    """Считает добавленные/удалённые строки между версиями (для превью)."""
    import difflib
    added = removed = 0
    for line in difflib.unified_diff(old.splitlines(), new.splitlines(), lineterm=""):
        if line.startswith("+") and not line.startswith("+++"):
            added += 1
        elif line.startswith("-") and not line.startswith("---"):
            removed += 1
    return added, removed


async def _selfedit_apply_draft(message: Message, uid: int) -> None:
    """Второй этап self-edit: бэкап → маркер авто-отката → запись черновика
    на диск → рестарт. Вызывается кнопкой «Применить»."""
    draft = selfedit_drafts.pop(uid, None)
    if draft is None:
        await message.edit_text(
            "😕 Черновик не найден — возможно, бот перезапускался. "
            "Сгенерируйте изменения заново."
        )
        return
    path = os.path.abspath(__file__)
    new_code, instruction, ts = draft["code"], draft["instruction"], draft["ts"]

    # бэкапим АКТУАЛЬНОЕ содержимое файла на момент применения
    backup = f"{path}.bak_{ts}"
    try:
        with open(path, encoding="utf-8") as f:
            current = f.read()
        with open(backup, "w", encoding="utf-8") as f:
            f.write(current)
    except Exception as e:
        await message.edit_text(f"😕 Не удалось сохранить бэкап: {html.quote(str(e))}")
        return

    try:
        with open(SELFUPDATE_MARKER, "w", encoding="utf-8") as f:
            json.dump(
                {"backup": backup, "attempts": 0, "uid": uid,
                 "instruction": instruction, "ts": ts},
                f, ensure_ascii=False,
            )
        with open(path, "w", encoding="utf-8") as f:
            f.write(new_code)
    except Exception as e:
        await message.edit_text(f"😕 Не удалось записать новый код: {html.quote(str(e))}")
        return

    await message.edit_text(
        "✅ Код записан на сервер, бэкап сохранён.\n"
        f"📦 <code>{html.quote(os.path.basename(backup))}</code>\n\n"
        "♻️ Перезапускаюсь с новым кодом… Если не выйду на связь за "
        f"{int(SELFUPDATE_HEALTH_DELAY)} с — сработает авто-откат."
    )
    logging.info("🛠 selfedit применён админом %s: %s", uid, instruction)
    await asyncio.sleep(1.0)   # даём Telegram доставить сообщение
    await _server_restart()    # панель/выход поднимет процесс заново с новым кодом


@router.callback_query(F.data == "admin:selfedit_apply")
async def cb_admin_selfedit_apply(cb: CallbackQuery) -> None:
    if not _is_admin(cb.from_user.id):
        await cb.answer("⛔", show_alert=True)
        return
    await cb.answer("Применяю…")
    await _selfedit_apply_draft(cb.message, cb.from_user.id)


@router.callback_query(F.data == "admin:selfedit_diff")
async def cb_admin_selfedit_diff(cb: CallbackQuery) -> None:
    if not _is_admin(cb.from_user.id):
        await cb.answer("⛔", show_alert=True)
        return
    draft = selfedit_drafts.get(cb.from_user.id)
    if draft is None:
        await cb.answer("Черновик не найден", show_alert=True)
        return
    import difflib
    diff_text = "\n".join(difflib.unified_diff(
        draft["source"].splitlines(), draft["code"].splitlines(),
        fromfile="текущий", tofile="черновик", lineterm="",
    ))
    doc = BufferedInputFile(
        diff_text.encode("utf-8"), filename=f"selfedit_{draft['ts']}.diff"
    )
    await cb.message.answer_document(doc, caption="📄 Diff черновика self-edit")
    await cb.answer()


@router.callback_query(F.data == "admin:selfedit_discard")
async def cb_admin_selfedit_discard(cb: CallbackQuery) -> None:
    if not _is_admin(cb.from_user.id):
        await cb.answer("⛔", show_alert=True)
        return
    selfedit_drafts.pop(cb.from_user.id, None)
    await cb.answer("Черновик удалён")
    await cb.message.edit_text(
        "❌ Черновик self-edit удалён — код на сервере не изменялся."
    )


@router.message(Command("rollback"))
async def cmd_rollback(msg: Message) -> None:
    """/rollback — вернуть последний бэкап исходника и перезапуститься (админ)."""
    uid = _uid(msg)
    if not _is_admin(uid):
        return
    path = os.path.abspath(__file__)
    backups = sorted(glob.glob(f"{path}.bak_*"))
    if not backups:
        await msg.answer("😕 Бэкапов не найдено.")
        return
    latest = backups[-1]
    try:
        with open(latest, encoding="utf-8") as f:
            good = f.read()
        compile(good, path, "exec")
        with open(path, "w", encoding="utf-8") as f:
            f.write(good)
    except Exception as e:
        await msg.answer(f"😕 Откат не удался: {html.quote(str(e))}")
        return
    try:
        if os.path.exists(SELFUPDATE_MARKER):
            os.remove(SELFUPDATE_MARKER)
    except OSError:
        pass
    await msg.answer(
        f"↩️ Откатился к бэкапу <code>{html.quote(os.path.basename(latest))}</code>.\n"
        "♻️ Перезапускаюсь…"
    )
    logging.info("🛠 rollback вручную админом %s → %s", uid, latest)
    await asyncio.sleep(1.0)
    await _server_restart()


async def _server_restart() -> None:
    """Перезапуск сервера. Если задан ключ панели play2go — через её API (как
    кнопка в консоли); иначе просто выходим и панель поднимает процесс сама."""
    save_state_now()
    if PANEL_API_KEY and http is not None:
        try:
            r = await http.post(
                f"{PANEL_URL}/api/client/servers/{SERVER_ID}/power",
                headers={
                    "Authorization": f"Bearer {PANEL_API_KEY}",
                    "Accept": "application/json",
                    "Content-Type": "application/json",
                },
                json={"signal": "restart"},
                timeout=30,
            )
            r.raise_for_status()
            logging.info("♻️ Рестарт через API панели отправлен.")
            return
        except Exception as e:
            logging.error("Рестарт через API панели не удался, выходим сами: %s", e)
    os._exit(0)


# ── Кнопки админ-панели: рестарт и изменение кода через Claude ──

@router.callback_query(F.data == "admin:restart")
async def cb_admin_restart(cb: CallbackQuery) -> None:
    if not _is_admin(cb.from_user.id):
        await cb.answer("⛔", show_alert=True)
        return
    kb = InlineKeyboardBuilder()
    kb.row(
        InlineKeyboardButton(text="♻️ Да, перезапустить", callback_data="admin:restart_do",
                             style="danger"),
        InlineKeyboardButton(text="⬅️ Отмена", callback_data="admin:menu",
                             style="primary"),
    )
    how = "через API панели" if PANEL_API_KEY else "выходом (панель поднимет сама)"
    await cb.message.edit_text(
        "♻️ <b>Перезапуск бота</b>\n\n"
        f"Способ: {how}.\n"
        "Бот на несколько секунд отключится и запустится заново. Продолжить?",
        reply_markup=kb.as_markup(),
    )
    await cb.answer()


@router.callback_query(F.data == "admin:restart_do")
async def cb_admin_restart_do(cb: CallbackQuery) -> None:
    if not _is_admin(cb.from_user.id):
        await cb.answer("⛔", show_alert=True)
        return
    await cb.answer("Перезапускаюсь…")
    await cb.message.edit_text("♻️ Перезапускаюсь… Вернусь через несколько секунд.")
    logging.info("♻️ Ручной рестарт админом %s", cb.from_user.id)
    await asyncio.sleep(1.0)
    await _server_restart()


@router.callback_query(F.data == "admin:selfedit")
async def cb_admin_selfedit(cb: CallbackQuery) -> None:
    if not _is_admin(cb.from_user.id):
        await cb.answer("⛔", show_alert=True)
        return
    if not ECHOGATE_KEY:
        await cb.answer("Не задан ECHOGATE_KEY в .env", show_alert=True)
        return
    admin_state[cb.from_user.id] = "waiting_selfedit"
    kb = InlineKeyboardBuilder()
    kb.row(InlineKeyboardButton(text="⬅️ Отмена", callback_data="admin:selfedit_cancel",
                                style="primary"))
    await cb.message.edit_text(
        "🛠 <b>Изменить код через AI</b>\n"
        "━━━━━━━━━━━━━━━\n"
        "Напишите одним сообщением, что изменить. Примеры:\n"
        "• <i>добавь команду /ping, отвечающую «pong»</i>\n"
        "• <i>измени приветствие в /start</i>\n"
        "━━━━━━━━━━━━━━━\n"
        f"🧠 <code>{html.quote(SELFEDIT_MODEL)}</code> · EchoGate\n"
        "📝 Сначала черновик и diff — на сервер код попадёт только "
        "после кнопки «Применить». При неудачном старте — авто-откат.",
        reply_markup=kb.as_markup(),
    )
    await cb.answer()


@router.callback_query(F.data == "admin:selfedit_cancel")
async def cb_admin_selfedit_cancel(cb: CallbackQuery) -> None:
    # Гвард обязателен, как и во всех admin:*: callback_data приходит от клиента,
    # и полагаться на то, что кнопку показали только админу, нельзя. Без него
    # любой, кто пришлёт эту строку, получал в свой чат отрисованную админ-панель.
    if not _is_admin(cb.from_user.id):
        await cb.answer("⛔", show_alert=True)
        return
    admin_state.pop(cb.from_user.id, None)
    await cb.answer("Отменено")
    await cb.message.edit_text(
        "🔧 <b>Админ-панель</b>\n━━━━━━━━━━━━━━━\n📊 аналитика · 👥 юзеры · 💰 финансы\n🛠 код · ⚙️ система\n\n<i>Выберите раздел:</i>",
        reply_markup=_admin_kb().as_markup(),
    )


def _selfupdate_startup_check() -> Optional[dict]:
    """Вызывается при старте. Если есть маркер незавершённого обновления —
    увеличивает счётчик попыток. Слишком много неудачных стартов подряд →
    авто-откат к бэкапу и рестарт. Возвращает данные маркера (для уведомления
    после выхода на связь) либо None."""
    if not os.path.exists(SELFUPDATE_MARKER):
        return None
    try:
        with open(SELFUPDATE_MARKER, encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        try:
            os.remove(SELFUPDATE_MARKER)
        except OSError:
            pass
        return None

    data["attempts"] = int(data.get("attempts", 0)) + 1
    if data["attempts"] > SELFUPDATE_MAX_ATTEMPTS:
        backup = data.get("backup", "")
        if backup and os.path.isfile(backup):
            try:
                with open(backup, encoding="utf-8") as f:
                    good = f.read()
                with open(os.path.abspath(__file__), "w", encoding="utf-8") as f:
                    f.write(good)
                logging.error(
                    "🛠 selfedit: АВТО-ОТКАТ к %s после %d неудачных стартов",
                    backup, data["attempts"] - 1,
                )
            except Exception as e:
                logging.error("🛠 selfedit: откат не удался: %s", e)
        try:
            os.remove(SELFUPDATE_MARKER)
        except OSError:
            pass
        os._exit(0)   # панель поднимет уже откачённый (рабочий) код

    try:
        with open(SELFUPDATE_MARKER, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)
    except Exception:
        pass
    return data


async def _selfupdate_confirm(bot: Bot, data: dict) -> None:
    """Через SELFUPDATE_HEALTH_DELAY сек стабильной работы считаем обновление
    успешным: удаляем маркер и сообщаем инициатору."""
    await asyncio.sleep(SELFUPDATE_HEALTH_DELAY)
    try:
        if os.path.exists(SELFUPDATE_MARKER):
            os.remove(SELFUPDATE_MARKER)
    except OSError:
        pass
    uid = data.get("uid")
    if not uid:
        return
    try:
        await bot.send_message(
            uid,
            "✅ Обновление применено успешно — бот работает на новом коде.\n"
            f"📦 Бэкап: <code>{html.quote(os.path.basename(data.get('backup', '')))}</code>\n"
            "↩️ Откат: <code>/rollback</code>",
        )
    except Exception:
        pass


def _admin_stats_text() -> str:
    total_users = len(user_stats)
    total_requests_ever = sum(s["total_requests"] for s in user_stats.values())
    today = date.today()
    active_today = sum(
        1 for uid, u in usage.items()
        if u.get("date") == today and u.get("used", 0) > 0
    )
    requests_today = sum(u.get("used", 0) for u in usage.values() if u.get("date") == today)
    stars_sold = sum(p["amount"] for p in purchases if p["currency"] == "XTR")
    rub_sold = sum(p["amount"] for p in purchases if p["currency"] == "RUB")

    cat_dist: dict[str, int] = {}
    for uid in user_stats:
        ck = user_category(uid)
        cat_dist[ck] = cat_dist.get(ck, 0) + 1

    top_categories = sorted(cat_dist.items(), key=lambda x: -x[1])[:5]
    top_str = "\n".join(
        f"  {CATEGORIES[k].emoji} {CATEGORIES[k].name}: <b>{c}</b>"
        for k, c in top_categories
    ) or "  —"

    return (
        "📊 <b>Статистика бота</b>\n\n"
        f"👥 Всего пользователей: <b>{total_users}</b>\n"
        f"🟢 Активны сегодня: <b>{active_today}</b>\n"
        f"💬 Потрачено токенов сегодня: <b>{fmt_tokens(requests_today)}</b>\n"
        f"📈 Всего сообщений: <b>{total_requests_ever}</b>\n"
        f"💵 Выручка: <b>{stars_sold}</b> ⭐ + <b>{rub_sold}</b> ₽\n\n"
        f"🏆 Популярные категории:\n{top_str}\n\n"
        f"<i>Обновлено: {datetime.now(timezone.utc).strftime('%d.%m.%Y %H:%M')} UTC</i>"
    )


def _admin_users_text(page: int = 0, per_page: int = 10) -> tuple[str, int]:
    all_users = sorted(user_stats.items(), key=lambda x: -x[1].get("total_requests", 0))
    total_pages = max(1, (len(all_users) + per_page - 1) // per_page)
    page = max(0, min(page, total_pages - 1))
    slice_ = all_users[page * per_page:(page + 1) * per_page]

    lines = [f"👥 <b>Пользователи</b> (стр. {page + 1}/{total_pages}):\n"]
    for uid, s in slice_:
        uname = f"@{s['username']}" if s.get("username") else s.get("full_name", str(uid))
        u = usage.get(uid, {})
        used_today = u.get("used", 0) if u.get("date") == date.today() else 0
        lines.append(
            f"• {html.quote(uname)} (<code>{uid}</code>)\n"
            f"  📊 всего: <b>{s['total_requests']}</b> | сегодня: <b>{used_today}</b>"
        )
    return "\n".join(lines), total_pages


def _admin_users_kb(page: int, total_pages: int) -> InlineKeyboardBuilder:
    b = InlineKeyboardBuilder()
    nav = []
    if page > 0:
        nav.append(InlineKeyboardButton(text="⬅️", callback_data=f"admin:users:{page - 1}"))
    if page < total_pages - 1:
        nav.append(InlineKeyboardButton(text="➡️", callback_data=f"admin:users:{page + 1}"))
    if nav:
        b.row(*nav)
    b.row(InlineKeyboardButton(text="⬅️ Назад", callback_data="admin:menu"))
    return b


@router.message(Command("admin"))
async def cmd_admin(msg: Message) -> None:
    uid = _uid(msg)
    if not _is_admin(uid):
        await msg.answer("⛔ Нет доступа.")
        return
    await msg.answer("🔧 <b>Админ-панель</b>\n━━━━━━━━━━━━━━━\n📊 аналитика · 👥 юзеры · 💰 финансы\n🛠 код · ⚙️ система\n\n<i>Выберите раздел:</i>", reply_markup=_admin_kb().as_markup())


@router.callback_query(F.data == "admin:project_close")
async def cb_admin_project_close(cb: CallbackQuery) -> None:
    if not _is_admin(cb.from_user.id):
        await cb.answer("⛔", show_alert=True)
        return
    global project_closed
    project_closed = not project_closed
    project_closed_notified.clear()  # тем, кого уже уведомили, можно снова показать при закрытии
    save_state_now()
    state = "ЗАКРЫТ" if project_closed else "ОТКРЫТ"
    await cb.answer(f"Проект {state}", show_alert=True)
    try:
        await cb.message.edit_text(
            "🔧 <b>Админ-панель</b>\n━━━━━━━━━━━━━━━\n📊 аналитика · 👥 юзеры · 💰 финансы\n🛠 код · ⚙️ система\n\n<i>Выберите раздел:</i>",
            reply_markup=_admin_kb().as_markup(),
        )
    except Exception:
        pass


@router.callback_query(F.data == "admin:menu")
async def cb_admin_menu(cb: CallbackQuery) -> None:
    if not _is_admin(cb.from_user.id):
        await cb.answer("⛔", show_alert=True)
        return
    await cb.answer()
    try:
        await cb.message.edit_text("🔧 <b>Админ-панель</b>\n━━━━━━━━━━━━━━━\n📊 аналитика · 👥 юзеры · 💰 финансы\n🛠 код · ⚙️ система\n\n<i>Выберите раздел:</i>", reply_markup=_admin_kb().as_markup())
    except Exception:
        pass


@router.callback_query(F.data == "admin:stats")
async def cb_admin_stats(cb: CallbackQuery) -> None:
    if not _is_admin(cb.from_user.id):
        await cb.answer("⛔", show_alert=True)
        return
    await cb.answer()
    b = InlineKeyboardBuilder()
    b.row(InlineKeyboardButton(text="🔄 Обновить", callback_data="admin:stats"))
    b.row(InlineKeyboardButton(text="⬅️ Назад", callback_data="admin:menu"))
    try:
        await cb.message.edit_text(_admin_stats_text(), reply_markup=b.as_markup())
    except Exception:
        pass


@router.callback_query(F.data.startswith("admin:users"))
async def cb_admin_users(cb: CallbackQuery) -> None:
    if not _is_admin(cb.from_user.id):
        await cb.answer("⛔", show_alert=True)
        return
    await cb.answer()
    parts = cb.data.split(":")
    page = int(parts[2]) if len(parts) > 2 else 0
    text, total = _admin_users_text(page)
    try:
        await cb.message.edit_text(text, reply_markup=_admin_users_kb(page, total).as_markup())
    except Exception:
        pass


@router.callback_query(F.data == "admin:broadcast")
async def cb_admin_broadcast(cb: CallbackQuery) -> None:
    if not _is_admin(cb.from_user.id):
        await cb.answer("⛔", show_alert=True)
        return
    await cb.answer()
    admin_state[cb.from_user.id] = "waiting_broadcast"
    b = InlineKeyboardBuilder()
    b.row(InlineKeyboardButton(text="❌ Отмена", callback_data="admin:broadcast_cancel"))
    try:
        await cb.message.edit_text(
            "📢 <b>Рассылка</b>\n\n"
            "Отправьте текст сообщения.\n\n"
            "Поддерживается Markdown-разметка:\n"
            "• <b>**жирный**</b> — <code>**жирный**</code>\n"
            "• <i>*курсив*</i> — <code>*курсив*</code>\n"
            "• <s>~~зачёркнутый~~</s> — <code>~~зачёркнутый~~</code>\n"
            "• <code>`код`</code> — <code>`код`</code>\n"
            "• <a href=\"https://example.com\">ссылка</a> — <code>[текст](url)</code>\n"
            "• <blockquote>цитата</blockquote> — <code>> цитата</code>\n\n"
            "HTML-теги тоже работают (<code>&lt;b&gt;</code>, <code>&lt;i&gt;</code> и т.д.).\n\n"
            "Сообщение получат все пользователи бота.",
            reply_markup=b.as_markup(),
        )
    except Exception:
        pass


@router.callback_query(F.data == "admin:give_all")
async def cb_admin_give_all(cb: CallbackQuery) -> None:
    if not _is_admin(cb.from_user.id):
        await cb.answer("⛔", show_alert=True)
        return
    await cb.answer()
    admin_state[cb.from_user.id] = "waiting_give_all"
    b = InlineKeyboardBuilder()
    b.row(InlineKeyboardButton(text="❌ Отмена", callback_data="admin:broadcast_cancel"))
    known = len((set(user_stats) | set(usage)) - set(ADMIN_IDS))
    try:
        await cb.message.edit_text(
            "➕ <b>Добавить токены ВСЕМ</b>\n\n"
            f"Известных боту пользователей: <b>{known}</b>.\n"
            "Отправьте число — начислю столько токенов каждому.\n"
            "Например: <code>10000</code>",
            reply_markup=b.as_markup(),
        )
    except Exception:
        pass


async def _admin_handle_give_all(msg: Message) -> None:
    """Начисляет указанное число токенов всем известным пользователям бота."""
    raw = (msg.text or "").strip().replace(" ", "")
    if not raw.isdigit() or int(raw) <= 0:
        await msg.answer(
            "😕 Нужно положительное число, например <code>10000</code>.\n"
            "Откройте админку и нажмите кнопку ещё раз.",
            reply_markup=_admin_kb().as_markup(),
        )
        return
    amount = int(raw)
    targets = (set(user_stats) | set(usage)) - set(ADMIN_IDS)
    for target_uid in targets:
        _grant_requests(target_uid, amount)
    save_state()
    await msg.answer(
        "✅ <b>Готово!</b>\n\n"
        f"➕ Начислено по <b>{fmt_tokens(amount)}</b> токенов\n"
        f"👥 Пользователей: <b>{len(targets)}</b>",
        reply_markup=_admin_kb().as_markup(),
    )


@router.callback_query(F.data == "admin:broadcast_send")
async def cb_admin_broadcast_send(cb: CallbackQuery) -> None:
    uid = cb.from_user.id
    if uid not in ADMIN_IDS:
        await cb.answer()
        return
    text = admin_pending_broadcast.pop(uid, None)
    if not text:
        await cb.answer("Нет текста для отправки", show_alert=True)
        return
    await cb.answer()
    try:
        await cb.message.edit_reply_markup(reply_markup=None)
    except Exception:
        pass
    await _run_broadcast(cb.message, cb.message.bot, text)


@router.callback_query(F.data == "admin:broadcast_cancel")
async def cb_admin_broadcast_cancel(cb: CallbackQuery) -> None:
    # Сначала гвард, потом любые изменения состояния — порядок важен как правило,
    # даже там, где ключ словаря совпадает с uid вызывающего и вреда нет.
    if not _is_admin(cb.from_user.id):
        await cb.answer("⛔", show_alert=True)
        return
    admin_pending_broadcast.pop(cb.from_user.id, None)
    admin_state.pop(cb.from_user.id, None)
    await cb.answer("Отменено")
    try:
        await cb.message.edit_text("🔧 <b>Админ-панель</b>\n━━━━━━━━━━━━━━━\n📊 аналитика · 👥 юзеры · 💰 финансы\n🛠 код · ⚙️ система\n\n<i>Выберите раздел:</i>", reply_markup=_admin_kb().as_markup())
    except Exception:
        pass


@router.callback_query(F.data == "admin:limits")
async def cb_admin_limits(cb: CallbackQuery) -> None:
    if not _is_admin(cb.from_user.id):
        await cb.answer("⛔", show_alert=True)
        return
    await cb.answer()
    b = InlineKeyboardBuilder()
    b.row(InlineKeyboardButton(text="⬅️ Назад", callback_data="admin:menu"))
    await cb.message.edit_text(
        f"⚙️ <b>Экономика бота</b>\n\n"
        f"💱 Курс: 1 ₽ = <b>{TOKENS_PER_RUB}</b> ткн · 1 ⭐ = <b>{TOKENS_PER_STAR}</b> ткн\n"
        f"💸 Мин. списание за ответ: <b>{fmt_tokens(MIN_TOKENS_SPEND)}</b>\n"
        f"🎁 Стартовый баланс новичку: <b>{fmt_tokens(WELCOME_TOKENS)}</b>\n"
        f"💎 Premium: −20% на списание\n"
        f"📜 История: <b>{MAX_HISTORY}</b> сообщений\n"
        f"🔤 Макс. токенов ответа: <b>{MAX_TOKENS}</b>\n\n"
        f"<i>Для изменения отредактируйте константы в коде и перезапустите бота.</i>",
        reply_markup=b.as_markup(),
    )


@router.callback_query(F.data == "admin:clear_all")
async def cb_admin_clear_all(cb: CallbackQuery) -> None:
    if not _is_admin(cb.from_user.id):
        await cb.answer("⛔", show_alert=True)
        return
    b = InlineKeyboardBuilder()
    b.row(
        InlineKeyboardButton(text="✅ Да, сбросить", callback_data="admin:clear_all_confirm"),
        InlineKeyboardButton(text="❌ Отмена", callback_data="admin:menu"),
    )
    await cb.answer()
    await cb.message.edit_text(
        "⚠️ <b>Вы уверены?</b>\n\nБудет удалена история всех пользователей. Это необратимо.",
        reply_markup=b.as_markup(),
    )


@router.callback_query(F.data == "admin:clear_all_confirm")
async def cb_admin_clear_all_confirm(cb: CallbackQuery) -> None:
    if not _is_admin(cb.from_user.id):
        await cb.answer("⛔", show_alert=True)
        return
    count = len(histories)
    histories.clear()
    await cb.answer(f"✅ Удалено историй: {count}", show_alert=True)
    await cb.message.edit_text(
        "🔧 <b>Админ-панель</b>\n━━━━━━━━━━━━━━━\n📊 аналитика · 👥 юзеры · 💰 финансы\n🛠 код · ⚙️ система\n\n<i>Выберите раздел:</i>",
        reply_markup=_admin_kb().as_markup(),
    )


# ═══════════════════════════════════════════════════════════════
# СОХРАНЕНИЕ СОСТОЯНИЯ (JSON) — переживает перезапуск
# ══════════════════════════════════════════════════════════════
# Сохраняем только важные данные: лимиты/покупки, Premium, выбранные
# категории, согласия и статистику. История диалогов (histories) и
# счётчики рекламы намеренно НЕ сохраняются — они эфемерны.

STATE_FILE = getenv("STATE_FILE", "bot_state.json")  # старый JSON — только для разовой миграции
DB_FILE = getenv("DB_FILE", "bot_state.db")          # основное хранилище (SQLite)


# Версия схемы БД. Поднимай на 1 при каждом изменении набора таблиц/колонок —
# записывается в PRAGMA user_version, чтобы по базе было видно, каким билдом
# она тронута. Сами миграции идут по факту (см. _apply_migrations), поэтому
# забытая версия схему не ломает.
SCHEMA_VERSION = 1


def _db_connect() -> sqlite3.Connection:
    """Открывает соединение с БД, создаёт таблицы и досоздаёт новые колонки."""
    conn = sqlite3.connect(DB_FILE)
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS usage(
            uid INTEGER PRIMARY KEY, date TEXT, used INTEGER, bought INTEGER);
        CREATE TABLE IF NOT EXISTS premium(
            uid INTEGER PRIMARY KEY, until TEXT);
        CREATE TABLE IF NOT EXISTS user_categories(
            uid INTEGER PRIMARY KEY, category TEXT);
        CREATE TABLE IF NOT EXISTS user_specific_model(
            uid INTEGER PRIMARY KEY, model TEXT);
        CREATE TABLE IF NOT EXISTS nvidia_consent(
            uid INTEGER PRIMARY KEY, consent INTEGER);
        CREATE TABLE IF NOT EXISTS agreement_accepted(
            uid INTEGER PRIMARY KEY, accepted INTEGER);
        CREATE TABLE IF NOT EXISTS captcha(
            uid INTEGER PRIMARY KEY);
        CREATE TABLE IF NOT EXISTS user_plans(
            uid INTEGER PRIMARY KEY, plan TEXT);
        CREATE TABLE IF NOT EXISTS user_stats(
            uid INTEGER PRIMARY KEY, data TEXT);
        CREATE TABLE IF NOT EXISTS op_state(
            uid INTEGER PRIMARY KEY, pass_date TEXT, free_used INTEGER);
        CREATE TABLE IF NOT EXISTS referrals(
            uid INTEGER PRIMARY KEY, invited_by INTEGER, count INTEGER);
        CREATE TABLE IF NOT EXISTS user_settings(
            uid INTEGER PRIMARY KEY, verbosity TEXT, persona TEXT, daily_bonus_date TEXT);
        CREATE TABLE IF NOT EXISTS platega_tx(
            tx_id TEXT PRIMARY KEY, uid INTEGER, kind TEXT, amount_rub INTEGER, credited INTEGER);
        CREATE TABLE IF NOT EXISTS banned(
            uid INTEGER PRIMARY KEY);
        CREATE TABLE IF NOT EXISTS purchases(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts TEXT, uid INTEGER, kind TEXT, title TEXT, amount INTEGER, currency TEXT);
        CREATE TABLE IF NOT EXISTS promo_codes(
            code TEXT PRIMARY KEY, data TEXT);
        CREATE TABLE IF NOT EXISTS trial_used(
            uid INTEGER PRIMARY KEY);
        CREATE TABLE IF NOT EXISTS welcome_granted(
            uid INTEGER PRIMARY KEY);
        CREATE TABLE IF NOT EXISTS kv_settings(
            key TEXT PRIMARY KEY, value TEXT);
        CREATE TABLE IF NOT EXISTS ref_milestones(
            uid INTEGER PRIMARY KEY, level INTEGER);
        CREATE TABLE IF NOT EXISTS user_prompts(
            uid INTEGER PRIMARY KEY, prompt TEXT);
        CREATE TABLE IF NOT EXISTS code_file_settings(
            uid INTEGER PRIMARY KEY, enabled INTEGER NOT NULL DEFAULT 0);
        CREATE TABLE IF NOT EXISTS bonus_streak(
            uid INTEGER PRIMARY KEY, streak INTEGER);
        CREATE TABLE IF NOT EXISTS ref_pending(
            uid INTEGER PRIMARY KEY, ref_uid INTEGER);
        CREATE TABLE IF NOT EXISTS lifetime_spent(
            uid INTEGER PRIMARY KEY, spent INTEGER);
        CREATE TABLE IF NOT EXISTS model_overrides(
            model_key TEXT PRIMARY KEY, state TEXT, data TEXT);
        CREATE TABLE IF NOT EXISTS model_descriptions(
            model_key TEXT PRIMARY KEY, description TEXT NOT NULL DEFAULT '');
        CREATE TABLE IF NOT EXISTS stars_charges(
            charge_id TEXT PRIMARY KEY, uid INTEGER DEFAULT 0,
            stars INTEGER DEFAULT 0, payload TEXT DEFAULT '',
            ts TEXT DEFAULT '', refunded INTEGER DEFAULT 0);
        """
    )
    _apply_migrations(conn)
    return conn


# Ожидаемый набор колонок по таблицам. CREATE TABLE IF NOT EXISTS выше НЕ
# трогает уже существующую таблицу со старой схемой, поэтому колонки,
# добавленные в новых версиях бота, надо досоздавать отдельно. Раньше это было
# сделано руками и только для stars_charges: любое новое поле в любой другой
# таблице ломало старт на старой базе с «no such column», потому что load_state
# селектит колонки явно.
#
# Как добавлять поле: дописываешь колонку сюда И в CREATE TABLE выше. Всё.
# Порядок и повторные запуски безопасны — ALTER выполняется только для
# реально отсутствующих колонок.
_EXPECTED_COLUMNS: dict[str, tuple[str, ...]] = {
    "usage": ("uid", "date", "used", "bought"),
    "premium": ("uid", "until"),
    "user_categories": ("uid", "category"),
    "user_specific_model": ("uid", "model"),
    "nvidia_consent": ("uid", "consent"),
    "agreement_accepted": ("uid", "accepted"),
    "captcha": ("uid",),
    "user_plans": ("uid", "plan"),
    "user_stats": ("uid", "data"),
    "op_state": ("uid", "pass_date", "free_used"),
    "referrals": ("uid", "invited_by", "count"),
    "user_settings": ("uid", "verbosity", "persona", "daily_bonus_date"),
    "platega_tx": ("tx_id", "uid", "kind", "amount_rub", "credited"),
    "banned": ("uid",),
    "purchases": ("id", "ts", "uid", "kind", "title", "amount", "currency"),
    "promo_codes": ("code", "data"),
    "trial_used": ("uid",),
    "welcome_granted": ("uid",),
    "kv_settings": ("key", "value"),
    "ref_milestones": ("uid", "level"),
    "user_prompts": ("uid", "prompt"),
    "code_file_settings": ("uid", "enabled"),
    "bonus_streak": ("uid", "streak"),
    "ref_pending": ("uid", "ref_uid"),
    "lifetime_spent": ("uid", "spent"),
    "model_overrides": ("model_key", "state", "data"),
    "model_descriptions": ("model_key", "description"),
    "stars_charges": ("charge_id", "uid", "stars", "payload", "ts", "refunded"),
}

# Тип и DEFAULT для досоздаваемых колонок. Ключ — "таблица.колонка", иначе
# берётся значение по имени колонки, иначе TEXT DEFAULT ''.
_COLUMN_TYPES: dict[str, str] = {
    "uid": "INTEGER DEFAULT 0",
    "used": "INTEGER DEFAULT 0",
    "bought": "INTEGER DEFAULT 0",
    "consent": "INTEGER DEFAULT 0",
    "accepted": "INTEGER DEFAULT 0",
    "free_used": "INTEGER DEFAULT 0",
    "invited_by": "INTEGER DEFAULT 0",
    "count": "INTEGER DEFAULT 0",
    "amount_rub": "INTEGER DEFAULT 0",
    "credited": "INTEGER DEFAULT 0",
    "amount": "INTEGER DEFAULT 0",
    "level": "INTEGER DEFAULT 0",
    "enabled": "INTEGER NOT NULL DEFAULT 0",
    "streak": "INTEGER DEFAULT 0",
    "ref_uid": "INTEGER DEFAULT 0",
    "spent": "INTEGER DEFAULT 0",
    "stars": "INTEGER DEFAULT 0",
    "refunded": "INTEGER DEFAULT 0",
    # Явно, иначе сработал бы фоллбэк TEXT DEFAULT '': на архаичной базе без
    # колонки id ALTER молча добавил бы текстовую, и ORDER BY id в выгрузке
    # покупок сортировал бы строками ('10' < '9'). PRIMARY KEY через ALTER
    # добавить нельзя, но тип хотя бы будет верный.
    "purchases.id": "INTEGER",
}


def _apply_migrations(conn: sqlite3.Connection) -> None:
    """Досоздаёт колонки, которых нет в существующих таблицах.

    Работает по факту (PRAGMA table_info), а не по номеру версии: старые базы
    без user_version тоже приводятся к актуальной схеме. Версия при этом
    выставляется — по ней видно, каким билдом база тронута последний раз.
    """
    added: list[str] = []
    for table, columns in _EXPECTED_COLUMNS.items():
        try:
            have = {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}
        except sqlite3.Error as e:
            logging.error(f"миграция: не читается схема {table}: {e}")
            continue
        if not have:
            continue  # таблицы нет вовсе — её только что создал CREATE TABLE
        for col in columns:
            if col in have:
                continue
            col_type = _COLUMN_TYPES.get(f"{table}.{col}") or _COLUMN_TYPES.get(col, "TEXT DEFAULT ''")
            try:
                conn.execute(f"ALTER TABLE {table} ADD COLUMN {col} {col_type}")
                added.append(f"{table}.{col}")
            except sqlite3.OperationalError as e:
                # PRIMARY KEY-колонку добавить нельзя; для таких случаев нужна
                # пересборка таблицы — сообщаем, но старт не блокируем.
                logging.warning(f"миграция: {table}.{col} не добавлена: {e}")
    if added:
        conn.commit()
        logging.info(f"миграция БД: добавлены колонки {', '.join(added)}")
    try:
        conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
    except sqlite3.Error as e:
        logging.debug(f"миграция: не удалось выставить user_version: {e}")


_persistent_state_conn: Optional[sqlite3.Connection] = None


def _state_conn() -> sqlite3.Connection:
    """Постоянное соединение для save_state с включённым WAL.

    Не пересоздаётся и не закрывается на каждый вызов — раньше save_state на
    КАЖДОЕ сообщение открывал новое соединение, прогонял весь CREATE TABLE и
    писал в медленном rollback-журнале. WAL + переиспользование соединения
    ускоряют коммит в разы. Схема данных и логика записи не меняются.
    """
    global _persistent_state_conn
    if _persistent_state_conn is None:
        c = _db_connect()  # создаёт схему, если её ещё нет
        try:
            c.execute("PRAGMA journal_mode=WAL")
            c.execute("PRAGMA synchronous=NORMAL")
        except Exception as e:
            logging.warning(f"save_state: не удалось включить WAL: {e}")
        _persistent_state_conn = c
    return _persistent_state_conn


# Флаг «состояние изменилось». save_state() больше НЕ пишет БД сразу:
# полная перезапись всех таблиц на каждое сообщение (O(все пользователи))
# блокировала event loop для всех. Теперь горячий путь только ставит флаг,
# а пишет _autosave_loop раз в 60с. Денежные операции по-прежнему пишутся
# немедленно через save_state_now().
_state_dirty: bool = False


def save_state() -> None:
    """Помечает состояние изменённым; на диск запишет автосейв (или save_state_now)."""
    global _state_dirty
    _state_dirty = True


def save_state_now() -> None:
    """Немедленно синхронизирует всё состояние в SQLite.

    Записи usage не за сегодня отбрасываются — так таблица не растёт со временем.
    """
    global _state_dirty, _persistent_state_conn
    try:
        today = date.today()
        conn = _state_conn()
        with conn:
            conn.execute("DELETE FROM usage")
            conn.executemany(
                "INSERT INTO usage(uid, date, used, bought) VALUES(?,?,?,?)",
                [
                    (uid, u["date"].isoformat(), u["used"], u["bought"])
                    for uid, u in usage.items()
                    # Пишем запись за сегодня ЛИБО любую с ненулевым платным
                    # балансом: bought — постоянные деньги пользователя, их
                    # нельзя терять из-за смены даты (иначе купленный пакет
                    # обнулялся после рестарта на следующий день).
                    if u.get("date") == today or int(u.get("bought", 0)) > 0
                ],
            )
            conn.execute("DELETE FROM premium")
            conn.executemany(
                "INSERT INTO premium(uid, until) VALUES(?,?)",
                [(uid, dt.isoformat()) for uid, dt in premium_until.items()],
            )
            conn.execute("DELETE FROM user_categories")
            conn.executemany(
                "INSERT INTO user_categories(uid, category) VALUES(?,?)",
                [(uid, c) for uid, c in user_categories.items()],
            )
            conn.execute("DELETE FROM user_specific_model")
            conn.executemany(
                "INSERT INTO user_specific_model(uid, model) VALUES(?,?)",
                [(uid, m) for uid, m in user_specific_model.items()],
            )
            conn.execute("DELETE FROM nvidia_consent")
            conn.executemany(
                "INSERT INTO nvidia_consent(uid, consent) VALUES(?,?)",
                [(uid, 1 if v else 0) for uid, v in nvidia_consent.items()],
            )
            conn.execute("DELETE FROM agreement_accepted")
            conn.executemany(
                "INSERT INTO agreement_accepted(uid, accepted) VALUES(?,?)",
                [(uid, 1 if v else 0) for uid, v in agreement_accepted.items()],
            )
            conn.execute("DELETE FROM captcha")
            conn.executemany(
                "INSERT INTO captcha(uid) VALUES(?)",
                [(uid,) for uid in captcha_solved],
            )
            conn.execute("DELETE FROM user_plans")
            conn.executemany(
                "INSERT INTO user_plans(uid, plan) VALUES(?,?)",
                [(uid, json.dumps(p, ensure_ascii=False)) for uid, p in user_plans.items()],
            )
            conn.execute("DELETE FROM user_stats")
            conn.executemany(
                "INSERT INTO user_stats(uid, data) VALUES(?,?)",
                [
                    (uid, json.dumps({**s, "first_seen": s["first_seen"].isoformat()}, ensure_ascii=False))
                    for uid, s in user_stats.items()
                ],
            )
            # Состояние ОП: дата последнего прохождения + израсходованные бесплатные ответы.
            conn.execute("DELETE FROM op_state")
            op_uids = set(op_pass_date) | set(op_free_used)
            conn.executemany(
                "INSERT INTO op_state(uid, pass_date, free_used) VALUES(?,?,?)",
                [
                    (uid, op_pass_date.get(uid), op_free_used.get(uid, 0))
                    for uid in op_uids
                ],
            )
            # Рефералы: кто кого пригласил + счётчик приглашённых.
            conn.execute("DELETE FROM referrals")
            ref_uids = set(invited_by) | set(referral_count)
            conn.executemany(
                "INSERT INTO referrals(uid, invited_by, count) VALUES(?,?,?)",
                [
                    (uid, invited_by.get(uid), referral_count.get(uid, 0))
                    for uid in ref_uids
                ],
            )
            # Персонализация: длина ответа, персона, дата ежедневного бонуса.
            conn.execute("DELETE FROM user_settings")
            set_uids = set(user_verbosity) | set(user_persona) | set(daily_bonus_date)
            conn.executemany(
                "INSERT INTO user_settings(uid, verbosity, persona, daily_bonus_date) VALUES(?,?,?,?)",
                [
                    (uid, user_verbosity.get(uid), user_persona.get(uid), daily_bonus_date.get(uid))
                    for uid in set_uids
                ],
            )
            # Платежи Platega (для идемпотентного зачисления после рестарта).
            conn.execute("DELETE FROM platega_tx")
            conn.executemany(
                "INSERT INTO platega_tx(tx_id, uid, kind, amount_rub, credited) VALUES(?,?,?,?,?)",
                [
                    (tx, r["uid"], r["kind"], r.get("amount_rub", 0), 1 if r.get("credited") else 0)
                    for tx, r in platega_tx.items()
                ],
            )
            # Баны, журнал покупок, промокоды, триал, скидка, реф-уровни, промпты.
            conn.execute("DELETE FROM banned")
            conn.executemany("INSERT INTO banned(uid) VALUES(?)", [(u,) for u in banned_users])
            conn.execute("DELETE FROM purchases")
            # id пишем явно и по порядку списка (он хронологический). На
            # архаичных базах, где id добавлен через ALTER, автоинкремента нет
            # и без этого все id были бы NULL — ORDER BY id терял хронологию.
            conn.executemany(
                "INSERT INTO purchases(id, ts, uid, kind, title, amount, currency) VALUES(?,?,?,?,?,?,?)",
                [(i, p["ts"], p["uid"], p["kind"], p["title"], p["amount"], p["currency"])
                 for i, p in enumerate(purchases, 1)],
            )
            conn.execute("DELETE FROM promo_codes")
            conn.executemany(
                "INSERT INTO promo_codes(code, data) VALUES(?,?)",
                [(c, json.dumps(i, ensure_ascii=False)) for c, i in promo_codes.items()],
            )
            conn.execute("DELETE FROM trial_used")
            conn.executemany("INSERT INTO trial_used(uid) VALUES(?)", [(u,) for u in trial_used])
            conn.execute("DELETE FROM welcome_granted")
            conn.executemany("INSERT INTO welcome_granted(uid) VALUES(?)",
                             [(u,) for u in welcome_granted])
            conn.execute("DELETE FROM kv_settings")
            if sale_info:
                conn.execute(
                    "INSERT INTO kv_settings(key, value) VALUES(?,?)",
                    ("sale", json.dumps(sale_info, ensure_ascii=False)),
                )
            # Пишем всегда, оба значения. Раньше строка появлялась только при
            # закрытии, а «открыт» выражался её отсутствием — из-за этого потеря
            # флага при загрузке (см. load_state) не просто игнорировалась, а
            # затирала закрытие: DELETE прошёл, INSERT не выполнился.
            conn.execute(
                "INSERT INTO kv_settings(key, value) VALUES(?,?)",
                ("project_closed", "1" if project_closed else "0"),
            )
            conn.execute("DELETE FROM ref_milestones")
            conn.executemany(
                "INSERT INTO ref_milestones(uid, level) VALUES(?,?)",
                list(ref_milestone_claimed.items()),
            )
            conn.execute("DELETE FROM user_prompts")
            conn.executemany(
                "INSERT INTO user_prompts(uid, prompt) VALUES(?,?)",
                list(user_custom_prompt.items()),
            )
            conn.execute("DELETE FROM code_file_settings")
            conn.executemany(
                "INSERT INTO code_file_settings(uid, enabled) VALUES(?,?)",
                [(uid, 1 if enabled else 0) for uid, enabled in user_code_files.items()],
            )
            conn.execute("DELETE FROM bonus_streak")
            conn.executemany(
                "INSERT INTO bonus_streak(uid, streak) VALUES(?,?)",
                list(daily_bonus_streak.items()),
            )
            conn.execute("DELETE FROM ref_pending")
            conn.executemany(
                "INSERT INTO ref_pending(uid, ref_uid) VALUES(?,?)",
                list(ref_pending_award.items()),
            )
            conn.execute("DELETE FROM lifetime_spent")
            conn.executemany(
                "INSERT INTO lifetime_spent(uid, spent) VALUES(?,?)",
                list(lifetime_spent.items()),
            )
            conn.execute("DELETE FROM stars_charges")
            conn.executemany(
                "INSERT INTO stars_charges(charge_id, uid, stars, payload, ts, refunded) "
                "VALUES(?,?,?,?,?,?)",
                [(c, int(p.get("uid", 0)), int(p.get("stars", 0)),
                  str(p.get("payload", "")), str(p.get("ts", "")),
                  1 if p.get("refunded") else 0)
                 for c, p in stars_charges.items()],
            )
        # Соединение постоянное (WAL) — НЕ закрываем, переиспользуем.
        # Флаг снимаем ТОЛЬКО после успешного коммита. Раньше он гасился в самом
        # начале функции: одна временная ошибка («database is locked», диск полон)
        # проглатывалась ниже, автосейв видел «чисто» и не повторял запись — всё
        # накопленное с прошлого удачного сохранения терялось молча.
        _state_dirty = False
    except Exception as e:
        logging.error(f"save_state error: {e}")
        # Состояние осталось несохранённым — пусть автосейв попробует снова.
        _state_dirty = True
        # Соединение могло умереть (закрыто, битый файл). Сбрасываем, чтобы
        # _state_conn() открыл новое: иначе каждый следующий сейв падал бы так же,
        # а бот продолжал работать как будто всё в порядке.
        try:
            if _persistent_state_conn is not None:
                _persistent_state_conn.close()
        except Exception:
            pass
        _persistent_state_conn = None


def _migrate_json_to_db() -> None:
    """Разовый перенос данных из старого bot_state.json в SQLite."""
    try:
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        logging.error(f"migrate: не удалось прочитать {STATE_FILE}: {e}")
        return

    for uid_s, u in data.get("usage", {}).items():
        try:
            usage[int(uid_s)] = {
                "date": date.fromisoformat(u["date"]),
                "used": int(u.get("used", 0)),
                "bought": int(u.get("bought", 0)),
            }
        except Exception:
            continue
    for uid_s, dt_s in data.get("premium_until", {}).items():
        try:
            premium_until[int(uid_s)] = datetime.fromisoformat(dt_s)
        except Exception:
            continue
    for uid_s, c in data.get("user_categories", {}).items():
        user_categories[int(uid_s)] = c
    for uid_s, m in data.get("user_specific_model", {}).items():
        user_specific_model[int(uid_s)] = m
    for uid_s, v in data.get("nvidia_consent", {}).items():
        nvidia_consent[int(uid_s)] = bool(v)
    for uid_s, p in data.get("user_plans", {}).items():
        user_plans[int(uid_s)] = p
    for uid_s, s in data.get("user_stats", {}).items():
        try:
            s2 = dict(s)
            s2["first_seen"] = datetime.fromisoformat(s["first_seen"])
            user_stats[int(uid_s)] = s2
        except Exception:
            continue

    # Бэкфилл welcome_granted здесь обязателен: load_state после миграции
    # выходит досрочно (return), и обычный бэкфилл при загрузке не выполняется.
    # Без этого все существующие пользователи получали стартовые токены повторно.
    # Стоит ДО save_state_now, чтобы бэкфилл гарантированно лёг в БД одной
    # записью с перенесёнными данными (после сейва он бы остался только в памяти).
    if not welcome_granted and user_stats:
        welcome_granted.update(user_stats.keys())
        logging.info(f"welcome_granted: бэкфилл {len(welcome_granted)} существующих пользователей (миграция JSON)")

    save_state_now()  # пишем перенесённые данные в БД
    # Убираем JSON в сторону ТОЛЬКО если БД действительно записалась.
    # Раньше os.replace выполнялся всегда: save_state_now глотает ошибки внутри
    # себя, поэтому при неудачной записи (нет прав, диск полон, битая БД) единственная
    # копия данных уезжала в .bak, а следующий старт поднимал бота пустым — и автосейв
    # закреплял эту пустоту. Теперь при сбое падаем, оставив JSON на месте.
    if not os.path.exists(DB_FILE) or _state_dirty:
        logging.critical(
            f"❌ Миграция в {DB_FILE} не удалась: база не создана или данные не записаны.\n"
            f"   {STATE_FILE} НЕ тронут — данные целы. Бот остановлен.\n"
            "   Проверьте права на папку и свободное место, затем запустите снова."
        )
        sys.exit(1)
    try:
        os.replace(STATE_FILE, STATE_FILE + ".bak")  # старый файл в бэкап, чтобы не мигрировать повторно
    except Exception as e:
        # Данные уже в БД, так что это не авария: при следующем старте DB_FILE
        # существует и ветка миграции не выполнится.
        logging.warning(f"миграция: не удалось переименовать {STATE_FILE}: {e}")
    logging.info(f"🔄 Данные перенесены из {STATE_FILE} в {DB_FILE}")


def load_state() -> None:
    """Загружает состояние из SQLite при старте (с разовой миграцией из JSON).

    global project_closed ОБЯЗАТЕЛЕН (добавлен 22.08.2026). Без него присваивание
    ниже создавало локальную переменную: признак закрытия читался из БД и молча
    выбрасывался, модульный флаг оставался False — после рестарта закрытый проект
    открывался всем. Хуже: через 60с автосейв делал DELETE FROM kv_settings и не
    возвращал строку обратно (она пишется только когда флаг истинный), так что
    закрытие стиралось насовсем. Единственная переменная в файле с такой ошибкой —
    проверено обходом AST по всем модульным скалярам.
    """
    global project_closed
    # Если БД ещё нет, но есть старый JSON — переносим его в БД
    if not os.path.exists(DB_FILE) and os.path.exists(STATE_FILE):
        _migrate_json_to_db()
        return  # словари уже заполнены в процессе миграции

    try:
        conn = _db_connect()
    except Exception as e:
        # ПАДАЕМ, а не продолжаем. Раньше здесь был return: словари оставались
        # пустыми, бот стартовал «чистым», а через 60с автосейв выполнял
        # DELETE FROM по всем таблицам и вписывал пустоту — молча уничтожая
        # балансы, Premium и рефералов всех пользователей. Не открылась БД
        # (повреждена, нет прав, занята другим процессом) — это авария,
        # требующая человека, а не повод работать без данных.
        logging.critical(
            f"❌ Не удалось открыть базу {DB_FILE}: {e}\n"
            "   Бот ОСТАНОВЛЕН, чтобы не перезаписать состояние пустыми данными.\n"
            "   Проверьте файл (целостность, права доступа) и восстановите из бэкапа."
        )
        sys.exit(1)

    today = date.today()
    for uid, d, used, bought in conn.execute("SELECT uid, date, used, bought FROM usage"):
        try:
            du = date.fromisoformat(d)
            if du != today and int(bought or 0) <= 0:
                continue  # устаревшая запись без платного баланса — пропускаем
            # Старые записи с bought > 0 загружаем: _get_usage сам обнулит
            # дневной used при первом обращении, а баланс сохранится.
            usage[int(uid)] = {"date": du, "used": int(used), "bought": int(bought)}
        except Exception:
            continue

    for uid, until in conn.execute("SELECT uid, until FROM premium"):
        try:
            premium_until[int(uid)] = datetime.fromisoformat(until)
        except Exception:
            continue

    for uid, c in conn.execute("SELECT uid, category FROM user_categories"):
        user_categories[int(uid)] = c

    for uid, m in conn.execute("SELECT uid, model FROM user_specific_model"):
        user_specific_model[int(uid)] = m

    for uid, v in conn.execute("SELECT uid, consent FROM nvidia_consent"):
        nvidia_consent[int(uid)] = bool(v)

    for uid, v in conn.execute("SELECT uid, accepted FROM agreement_accepted"):
        agreement_accepted[int(uid)] = bool(v)

    for (uid,) in conn.execute("SELECT uid FROM captcha"):
        captcha_solved.add(int(uid))

    for uid, p in conn.execute("SELECT uid, plan FROM user_plans"):
        try:
            user_plans[int(uid)] = json.loads(p)
        except Exception:
            continue

    for uid, dat in conn.execute("SELECT uid, data FROM user_stats"):
        try:
            s = json.loads(dat)
            s["first_seen"] = datetime.fromisoformat(s["first_seen"])
            user_stats[int(uid)] = s
        except Exception:
            continue

    for uid, pass_date, free_used in conn.execute(
            "SELECT uid, pass_date, free_used FROM op_state"):
        try:
            if pass_date:
                op_pass_date[int(uid)] = pass_date
            op_free_used[int(uid)] = int(free_used or 0)
        except Exception:
            continue

    for uid, inv_by, count in conn.execute(
            "SELECT uid, invited_by, count FROM referrals"):
        try:
            if inv_by is not None:
                invited_by[int(uid)] = int(inv_by)
            referral_count[int(uid)] = int(count or 0)
        except Exception:
            continue

    for uid, verbosity, persona, bonus_date in conn.execute(
            "SELECT uid, verbosity, persona, daily_bonus_date FROM user_settings"):
        try:
            if verbosity:
                user_verbosity[int(uid)] = verbosity
            if persona:
                user_persona[int(uid)] = persona
            if bonus_date:
                daily_bonus_date[int(uid)] = bonus_date
        except Exception:
            continue

    for tx, uid, kind, amount_rub, credited in conn.execute(
            "SELECT tx_id, uid, kind, amount_rub, credited FROM platega_tx"):
        try:
            platega_tx[str(tx)] = {
                "uid": int(uid), "kind": kind,
                "amount_rub": int(amount_rub or 0), "credited": bool(credited),
            }
        except Exception:
            continue

    for (uid,) in conn.execute("SELECT uid FROM banned"):
        banned_users.add(int(uid))

    for ts, uid, kind, title, amount, currency in conn.execute(
            "SELECT ts, uid, kind, title, amount, currency FROM purchases ORDER BY id"):
        purchases.append({"ts": ts, "uid": int(uid), "kind": kind,
                          "title": title, "amount": int(amount or 0), "currency": currency})

    for code, dat in conn.execute("SELECT code, data FROM promo_codes"):
        try:
            promo_codes[str(code)] = json.loads(dat)
        except Exception:
            continue

    for (uid,) in conn.execute("SELECT uid FROM trial_used"):
        trial_used.add(int(uid))
    for (uid,) in conn.execute("SELECT uid FROM welcome_granted"):
        welcome_granted.add(int(uid))
    # Разовый бэкфилл для баз, созданных до появления welcome_granted: все, кто
    # уже есть в user_stats, стартовые токены очевидно получили. Без этого их
    # первая же чистка по неактивности выдала бы подарок повторно.
    if not welcome_granted and user_stats:
        welcome_granted.update(user_stats.keys())
        logging.info(f"welcome_granted: бэкфилл {len(welcome_granted)} существующих пользователей")
        save_state()

    for key, value in conn.execute("SELECT key, value FROM kv_settings"):
        if key == "sale":
            try:
                # Замена, а не merge: раньше update сохранял ключи старой
                # акции (percent/until) и «отменённая» распродажа воскресала
                # после рестарта с частично старыми параметрами.
                sale_info.clear()
                sale_info.update(json.loads(value))
            except Exception:
                pass
        elif key == "project_closed":
            project_closed = str(value).strip() in ("1", "true", "True")

    for uid, level in conn.execute("SELECT uid, level FROM ref_milestones"):
        ref_milestone_claimed[int(uid)] = int(level or 0)

    for uid, prompt in conn.execute("SELECT uid, prompt FROM user_prompts"):
        if prompt:
            user_custom_prompt[int(uid)] = prompt

    for uid, enabled in conn.execute("SELECT uid, enabled FROM code_file_settings"):
        user_code_files[int(uid)] = bool(enabled)

    for uid, streak in conn.execute("SELECT uid, streak FROM bonus_streak"):
        daily_bonus_streak[int(uid)] = int(streak or 0)

    for uid, r_uid in conn.execute("SELECT uid, ref_uid FROM ref_pending"):
        if r_uid is not None:
            ref_pending_award[int(uid)] = int(r_uid)

    for uid, spent in conn.execute("SELECT uid, spent FROM lifetime_spent"):
        lifetime_spent[int(uid)] = int(spent or 0)

    for charge_id, p_uid, p_stars, p_payload, p_ts, p_ref in conn.execute(
            "SELECT charge_id, uid, stars, payload, ts, refunded FROM stars_charges"):
        stars_charges[str(charge_id)] = {
            "uid": int(p_uid or 0), "stars": int(p_stars or 0),
            "payload": str(p_payload or ""), "ts": str(p_ts or ""),
            "refunded": bool(p_ref),
        }

    conn.close()
    logging.info(
        f"📂 Состояние загружено из SQLite: {len(usage)} usage, "
        f"{len(premium_until)} premium, {len(user_stats)} stats"
    )


async def _autosave_loop(interval: int = 60) -> None:
    """Периодически сохраняет состояние на диск (только если что-то менялось).

    Тело обёрнуто в try/except намеренно: раньше любое исключение внутри
    (например, заблокированная БД) убивало задачу насмерть. Бот при этом
    продолжал обслуживать людей, но на диск не писал НИЧЕГО до перезапуска,
    и узнать об этом было нельзя. Теперь сбой одной итерации логируется, а
    цикл живёт дальше.
    """
    while True:
        await asyncio.sleep(interval)
        try:
            if _state_dirty:
                save_state_now()
        except Exception as e:
            logging.error(f"autosave: сохранение не удалось, повтор через {interval}с: {e}")


# Ссылка на задачу остановки polling (создаётся в обработчике сигнала).
# Нужна как раз для того, чтобы GC не собрал её до выполнения.
_shutdown_task: Optional[asyncio.Task] = None


def _watch_task(task: asyncio.Task, name: str) -> asyncio.Task:
    """Логирует падение фоновой задачи вместо тихой смерти.

    Без этого исключение внутри create_task никуда не попадает: задача мертва,
    бот работает, симптомов нет. Ставится на все фоновые циклы.
    """
    def _done(t: asyncio.Task) -> None:
        if t.cancelled():
            return
        exc = t.exception()
        if exc is not None:
            logging.error(f"фоновая задача «{name}» упала: {exc!r}", exc_info=exc)

    task.add_done_callback(_done)
    return task


# Час ежедневного автобэкапа по UTC (время сервера). По умолчанию 4:00 UTC =
# 07:00 по Москве (МСК = UTC+3, круглый год без перевода часов).
# Переопределяется через переменную окружения BACKUP_HOUR_UTC, без правки кода.
BACKUP_HOUR_UTC: int = int(getenv("BACKUP_HOUR_UTC", "4"))


async def _daily_backup_loop(bot: Bot) -> None:
    """Каждый день в BACKUP_HOUR_UTC:00 шлёт админам бэкап: БД, код, .env.

    Страховка на случай, если хостинг закончится/умрёт: у админа в Telegram
    всегда лежит вчерашняя копия всего, что нужно для переезда.
    """
    while True:
        # Сколько спать до ближайших HH:00 по UTC.
        now = datetime.now(timezone.utc)
        target = now.replace(hour=BACKUP_HOUR_UTC, minute=0, second=0, microsecond=0)
        if target <= now:
            target += timedelta(days=1)
        await asyncio.sleep((target - now).total_seconds())

        save_state_now()  # свежие данные в БД перед отправкой
        # WAL: свежие транзакции лежат в bot_state.db-wal, который в бэкап не
        # входит. Чекпоинт сгоняет всё в основной файл, иначе «страховая копия»
        # для переезда может не содержать последние платежи и балансы.
        try:
            _state_conn().execute("PRAGMA wal_checkpoint(TRUNCATE)")
        except Exception as e:
            logging.warning(f"Автобэкап: wal_checkpoint не удался: {e}")
        stamp = datetime.now(timezone.utc).strftime("%d.%m.%Y %H:%M UTC")
        files = [
            (DB_FILE, f"💾 База данных · автобэкап {stamp}"),
            (os.path.abspath(__file__), f"📄 Код бота · автобэкап {stamp}"),
        ]
        # .env В АВТОБЭКАП НЕ ВХОДИТ (22.08.2026). Раньше он уезжал сюда каждый
        # день: загруженный в Telegram файл остаётся в облаке навсегда и доступен
        # по file_id даже после удаления сообщения, то есть все токены бота
        # копились в переписке в открытом виде. Одна опечатка в ADMIN_IDS — и они
        # ежедневно уходят постороннему.
        # Смысл автобэкапа (переезд, если умрёт хостинг) сохранён: БД и код на
        # месте, а секреты нужно один раз положить в менеджер паролей. Если очень
        # нужно вернуть старое поведение — BACKUP_ENV=1 в .env, осознанно.
        if getenv("BACKUP_ENV", "0") == "1":
            env_path = ".env" if os.path.isfile(".env") else \
                os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
            files.append((env_path, f"⚙️ .env · автобэкап {stamp} — внутри все секреты!"))
        for admin_id in ADMIN_IDS:
            for path, caption in files:
                try:
                    if not os.path.isfile(path):
                        await bot.send_message(admin_id, f"⚠️ Автобэкап: файл не найден: <code>{html.quote(path)}</code>")
                        continue
                    await bot.send_document(admin_id, FSInputFile(path), caption=caption)
                except Exception as e:
                    logging.warning(f"Автобэкап: не удалось отправить {path} админу {admin_id}: {e}")


# ── Ежедневная новостная сводка админам ──────────────────────
# Час по UTC. По умолчанию 8:00 UTC = 11:00 по Москве (МСК = UTC+3).
NEWS_HOUR_UTC: int = int(getenv("NEWS_HOUR_UTC", "8"))
# Последний день рассылки (включительно). После него цикл сам останавливается.
NEWS_UNTIL: date = date.fromisoformat(getenv("NEWS_UNTIL", "2026-08-31"))
# Бесплатная модель для суммаризации (сами новости берём из RSS, а не веб-поиском).
NEWS_MODEL: str = getenv("NEWS_MODEL", "inclusionai/ling-3.0-flash:free")
# RSS-ленты, из которых собираются новости (через запятую в .env).
NEWS_FEEDS: list[str] = [
    u.strip() for u in getenv(
        "NEWS_FEEDS",
        "https://lenta.ru/rss/news,"
        "https://habr.com/ru/rss/news/?fl=ru,"
        "https://3dnews.ru/news/rss/",
    ).split(",") if u.strip()
]


async def _fetch_news_items(max_per_feed: int = 15) -> list[dict]:
    """Скачивает RSS-ленты и возвращает свежие (за ~26 часов) новости.

    Бесплатно и без ключей: обычные HTTP-запросы к открытым RSS.
    Каждый элемент: {"title", "link", "source"}.
    """
    import xml.etree.ElementTree as ET
    from email.utils import parsedate_to_datetime

    cutoff = datetime.now(timezone.utc) - timedelta(hours=26)
    items: list[dict] = []
    for feed_url in NEWS_FEEDS:
        try:
            r = await http.get(feed_url, timeout=20,
                               headers={"User-Agent": "Mozilla/5.0 (news-digest-bot)"})
            r.raise_for_status()
            root = ET.fromstring(r.content)
            source = (root.findtext("channel/title") or feed_url)[:40]
            count = 0
            for it in root.iterfind("channel/item"):
                title = (it.findtext("title") or "").strip()
                link = (it.findtext("link") or "").strip()
                pub = (it.findtext("pubDate") or "").strip()
                if not title or not link:
                    continue
                if pub:  # старые записи пропускаем
                    try:
                        if parsedate_to_datetime(pub) < cutoff:
                            continue
                    except Exception:
                        pass
                items.append({"title": title, "link": link, "source": source})
                count += 1
                if count >= max_per_feed:
                    break
        except Exception as e:
            logging.warning("Новостная рассылка: лента %s недоступна: %s", feed_url, e)
    return items


async def _send_news_digest(bot: Bot) -> None:
    """Собирает сводку новостей и шлёт её всем админам.

    Используется ежедневным циклом и кнопкой «📰 Тест сводки» в админке.
    """
    today_str = datetime.now(timezone.utc).strftime("%d.%m.%Y")
    news = await _fetch_news_items()
    if not news:
        for admin_id in ADMIN_IDS:
            try:
                await bot.send_message(
                    admin_id, "⚠️ Сводка новостей: ни одна RSS-лента не ответила.")
            except Exception:
                pass
        return

    listing = "\n".join(
        f"- [{n['source']}] {n['title']} — {n['link']}" for n in news
    )
    prompt = (
        f"Сегодня {today_str}. Ниже список свежих новостей из RSS-лент.\n"
        "Выбери 6–10 самых важных и составь сводку на русском: каждый пункт — "
        "заголовок, одно предложение сути своими словами и ссылка из списка. "
        "Без вступлений и выводов, только пункты. Используй только новости из "
        f"списка, ничего не выдумывай.\n\n{listing}"
    )
    try:
        text = await _call_model(
            NEWS_MODEL,
            [{"role": "user", "content": prompt}],
            temperature=0.3,
        )
    except Exception as e:
        # Free-модель может быть перегружена — тогда шлём сырые заголовки.
        logging.warning("Новостная рассылка: ошибка модели %s: %s", NEWS_MODEL, e)
        text = "Модель недоступна, вот заголовки без сводки:\n\n" + listing

    # Отправляем как обычный текст (без HTML): модель отвечает маркдауном,
    # и символы вроде "<" сломали бы HTML-парсер Telegram.
    full = f"📰 Сводка новостей · {today_str}\n\n{text}"
    chunks = [full[i:i + MAX_TG] for i in range(0, len(full), MAX_TG)]
    for admin_id in ADMIN_IDS:
        try:
            for chunk in chunks:
                await bot.send_message(admin_id, chunk, parse_mode=None)
        except Exception as e:
            logging.warning("Новостная рассылка: не отправилось %s: %s", admin_id, e)


async def _daily_news_loop(bot: Bot) -> None:
    """Каждый день в NEWS_HOUR_UTC:00 шлёт админам новостную сводку.

    Работает до NEWS_UNTIL включительно, после — тихо завершается.
    Схема полностью бесплатная: новости собираются из открытых RSS-лент,
    а короткую сводку по ним пишет free-модель OpenRouter. Если модель
    недоступна — придёт просто список заголовков со ссылками.
    """
    while True:
        now = datetime.now(timezone.utc)
        # Если уже прошли дедлайн — завершаемся
        if now.date() > NEWS_UNTIL:
            logging.info("Новостная рассылка: %s прошло, цикл остановлен", NEWS_UNTIL)
            return
        target = now.replace(hour=NEWS_HOUR_UTC, minute=0, second=0, microsecond=0)
        if target <= now:
            # Час уже прошёл сегодня. Если это последний день — отправляем сейчас,
            # чтобы не потерять сводку (иначе target уйдёт на завтра > NEWS_UNTIL).
            if now.date() == NEWS_UNTIL:
                try:
                    await _send_news_digest(bot)
                except Exception as e:
                    logging.error(f"последняя новостная сводка не отправлена: {e}")
                return
            target += timedelta(days=1)
        await asyncio.sleep((target - now).total_seconds())
        # Ошибку глотаем: одна неудачная сводка (RSS не ответил, модель упала)
        # раньше убивала цикл целиком, и рассылки не было до перезапуска бота.
        try:
            await _send_news_digest(bot)
        except Exception as e:
            logging.error(f"новостная сводка не отправлена, жду следующего дня: {e}")


async def _cleanup_loop(interval: int = 3600) -> None:
    """Раз в час запускает уборку. Ошибку одного прохода логирует и продолжает.

    Без try/except первая же ошибка внутри убивала задачу навсегда: истёкший
    Premium перестал бы сниматься, а данные неактивных — удаляться, и заметить
    это было бы нечем (задача просто исчезала).
    """
    while True:
        await asyncio.sleep(interval)
        try:
            _cleanup_once()
        except Exception as e:
            logging.error(f"уборка не выполнилась, повтор через {interval}с: {e}", exc_info=e)


def _cleanup_once() -> None:
    """Один проход уборки: истёкший Premium + данные неактивных пользователей."""
    now = datetime.now(timezone.utc)
    today = date.today()

    def _charge_age_days(ts: str) -> int:
        try:
            return (now - datetime.fromisoformat(ts)).days
        except Exception:
            return 0  # битую дату не удаляем — разберёмся руками

    # Истёкший Premium
    expired = [uid for uid, dt in premium_until.items() if dt < now]
    for uid in expired:
        del premium_until[uid]

    # Чистка леджера Stars: он нужен для возвратов и защиты от повторной
    # доставки successful_payment (Telegram переотправляет её недолго).
    # Без чистки таблица росла вечно — по записи на каждый платёж.
    stale_charges = [
        cid for cid, p in stars_charges.items()
        if p.get("ts") and _charge_age_days(p["ts"]) > 90
    ]
    for cid in stale_charges:
        del stars_charges[cid]

    # Неактивные пользователи (не было запросов >7 дней)
    inactive = [
        uid for uid, u in usage.items()
        if isinstance(u.get("date"), date) and (today - u["date"]).days > 7
        and int(u.get("bought", 0)) <= 0  # с деньгами на балансе не трогаем
    ]
    for uid in inactive:
        histories.pop(uid, None)
        usage.pop(uid, None)
        user_categories.pop(uid, None)
        user_specific_model.pop(uid, None)
        nvidia_consent.pop(uid, None)
        botohub_msg_counter.pop(uid, None)
        op_pass_date.pop(uid, None)
        op_free_used.pop(uid, None)
        pending_ai.pop(uid, None)
        pending_auto_switch.pop(uid, None)
        user_stats.pop(uid, None)
        user_verbosity.pop(uid, None)
        user_code_files.pop(uid, None)
        daily_bonus_date.pop(uid, None)
        pending_referral.pop(uid, None)
        user_input_state.pop(uid, None)
        captcha_solved.discard(uid)
        banned_notified.discard(uid)
        daily_bonus_streak.pop(uid, None)
        last_ai_request_at.pop(uid, None)
        # Персону и личный промпт у активного Premium не трогаем: это оплаченная
        # настройка, а premium_until не чистится — Premium выживет, а его
        # персонализация пропала бы.
        if not premium_active(uid):
            user_persona.pop(uid, None)
            user_custom_prompt.pop(uid, None)
        # lifetime_spent не чистим, пока по человеку висит невыданная награда
        # рефереру: счётчик — это прогресс до порога активации, и его обнуление
        # заставляло приглашённого нарабатывать порог заново.
        if uid not in ref_pending_award:
            lifetime_spent.pop(uid, None)
            # user_msg_to_bot_msg ключуется по chat_id: в личке он равен uid,
            # поэтому неактивные личные чаты вычищаются. Групповые chat_id так не
            # достать — внутренние словари там ограничены MSG_MAP_LIMIT, но сами
            # ключи чатов остаются. Для бота, работающего в личке, этого хватает.
            user_msg_to_bot_msg.pop(uid, None)
            # agreement_accepted, trial_used, invited_by и welcome_granted
            # СОЗНАТЕЛЬНО не чистим: это постоянные метки, а не кэш. Их удаление
            # открывало ферму с периодом 7 дней — юзер сжигал подарочные токены,
            # ждал неделю, писал снова и получал заново и триал Premium, и награду
            # рефереру. welcome_granted добавлен потому, что стартовые токены
            # раньше проверялись по «uid нет в user_stats», а user_stats как раз
            # удаляется строкой выше — то есть подарок выдавался повторно.
            # Экономия памяти тут пара байт на uid, цена ошибки — подарки без дна.

    # ВНЕ цикла: раньше save_state стоял внутри for по inactive и не выполнялся,
    # когда истёк только Premium (inactive пуст) — удаление не попадало в БД,
    # рестарт «воскрешал» протухший Premium.
    if expired or inactive:
        save_state()


# ══════════════════════════════════════════════════════════════
# ЗАПУСК
# ══════════════════════════════════════════════════════════════

async def main() -> None:
    global http
    # Авто-откат: если прошлое /selfedit-обновление «окирпичило» бота (падение
    # при старте несколько раз подряд) — вернём бэкап ещё до всего остального.
    _selfupdate_pending = _selfupdate_startup_check()

    bot = Bot(token=BOT_TOKEN, default=DefaultBotProperties(parse_mode=ParseMode.HTML))
    http = httpx.AsyncClient()

    if _selfupdate_pending is not None:
        # Ссылку держим в переменной: задача без сильной ссылки может быть собрана
        # сборщиком мусора на середине. _watch_task заодно покажет её падение в
        # логе, а не проглотит.
        selfupdate_task = _watch_task(
            asyncio.create_task(_selfupdate_confirm(bot, _selfupdate_pending)),
            "selfupdate_confirm",
        )

    load_state()
    autosave_task = _watch_task(asyncio.create_task(_autosave_loop()), "autosave")
    cleanup_task = _watch_task(asyncio.create_task(_cleanup_loop()), "cleanup")
    backup_task = _watch_task(asyncio.create_task(_daily_backup_loop(bot)), "backup")
    news_task = _watch_task(asyncio.create_task(_daily_news_loop(bot)), "news")

    # Корректное завершение по сигналу. Без этого docker stop / systemd restart /
    # деплой убивали процесс мгновенно: finally с save_state_now() не отрабатывал
    # и терялось всё, накопленное с последнего автосейва (до 60с списаний,
    # стриков, реф-наград) — то есть при КАЖДОМ обычном рестарте.
    # add_signal_handler есть только на Unix; на Windows тихо пропускаем —
    # там работает KeyboardInterrupt и finally в конце main().
    def _on_signal(sig_name: str) -> None:
        logging.info(f"⏹ Получен {sig_name} — сохраняю состояние и останавливаюсь")
        try:
            save_state_now()
        except Exception as e:
            logging.error(f"{sig_name}: сохранить состояние не удалось: {e}")
        # Снимаем polling: dp.stop_polling() завершает start_polling штатно,
        # управление уходит в finally, где закроются http-клиент и сессия бота.
        # Ссылку кладём в глобальную переменную: локальная умрёт вместе с
        # _on_signal, и задачу может собрать GC до того, как она выполнится.
        global _shutdown_task
        _shutdown_task = asyncio.create_task(dp.stop_polling())

    try:
        import signal as _signal
        loop = asyncio.get_running_loop()
        for _sig in (_signal.SIGTERM, _signal.SIGINT):
            loop.add_signal_handler(_sig, _on_signal, _sig.name)
    except (ImportError, NotImplementedError, AttributeError, RuntimeError) as e:
        logging.debug(f"обработчики сигналов недоступны на этой платформе: {e}")

    me = await bot.get_me()
    rich_status = "✅" if TELEGRAMIFY_AVAILABLE else "⚠️ нет telegramify-markdown, fallback на plain text"
    admin_status = f"👮 Admins: {ADMIN_IDS}" if ADMIN_IDS else "⚠️ ADMIN_IDS не задан"
    logging.info(
        f"🤖 @{me.username} | "
        f"🧠 OpenRouter ({len(CATEGORIES)} категорий, {len(MODELS)} моделей) | "
        f"🎤 Voxtral | "

        f"⭐ Stars | "
        f"📺 BotoHub Views (каждые {BOTOHUB_AD_EVERY} сообщений) | "
        f"🔒 ОП {'вкл' if BOTOHUB_OP_TOKEN else 'выкл (нет BOTOHUB_OP_TOKEN)'} | "
        f"📄 Rich Messages {rich_status} | "
        f"{admin_status}"
    )

    # drop_pending_updates=False: среди накопившихся апдейтов могут быть
    # successful_payment (Telegram Stars) — если их дропнуть, пользователь
    # заплатил, а токены не начислятся. Повторные доставки платежей теперь
    # отсекаются леджером stars_charges, так что дубли не страшны.
    await bot.delete_webhook(drop_pending_updates=False)
    try:
        await dp.start_polling(bot)
    except TelegramConflictError:
        logging.error(
            "⛔ Конфликт: уже запущен другой экземпляр этого бота с тем же "
            "BOT_TOKEN. Проверьте задачи Python и завершите дубликаты."
        )
        raise
    finally:
        autosave_task.cancel()
        cleanup_task.cancel()
        backup_task.cancel()
        news_task.cancel()
        save_state_now()   # финальное сохранение при остановке
        await http.aclose()
        await bot.session.close()


# ══════════════════════════════════════════════════════════════
# АДМИН: УПРАВЛЕНИЕ МОДЕЛЯМИ (техработы / скрытие / добавление)
# ══════════════════════════════════════════════════════════════

def _danger_button(text: str, callback_data: str) -> InlineKeyboardButton:
    """Красная кнопка в новых Bot API; безопасный fallback для старых версий."""
    try:
        return InlineKeyboardButton(text=text, callback_data=callback_data, style="danger")
    except Exception:
        return InlineKeyboardButton(text=text, callback_data=callback_data)


def _admin_models_menu_kb() -> InlineKeyboardBuilder:
    b = InlineKeyboardBuilder()
    b.row(InlineKeyboardButton(text="🔧 Техработы (вкл/выкл)", callback_data="admin:models:mode:m"))
    b.row(InlineKeyboardButton(text="🗑 Скрыть / вернуть", callback_data="admin:models:mode:h"))
    b.row(InlineKeyboardButton(text="📝 Описание модели", callback_data="admin:models:mode:d"))
    b.row(InlineKeyboardButton(text="➕ Добавить модель", callback_data="admin:models:add"))
    b.row(_danger_button("🗑 Удалить модель", "admin:models:mode:x"))
    b.row(InlineKeyboardButton(text="⬅️ В админку", callback_data="admin:menu"))
    return b


def _admin_category_model_keys(cat_key: str) -> list[str]:
    _sync_custom_models()
    overrides = _model_overrides()
    keys = list(CATEGORIES[cat_key].models)
    keys += [
        k for k, row in overrides.items()
        if _is_custom_override(k, row) and row["data"].get("category") == cat_key
    ]
    return [
        k for k in dict.fromkeys(keys)
        if k in MODELS and overrides.get(k, {}).get("state") != "deleted"
    ]


def _admin_models_list_kb(mode: str, cat_key: str, page: int) -> InlineKeyboardBuilder:
    """Клавиатура моделей категории; тап переключает состояние модели."""
    overrides = _model_overrides()
    keys = _admin_category_model_keys(cat_key)
    page_count = max(1, (len(keys) + 5) // 6)
    page = max(0, min(page, page_count - 1))
    b = InlineKeyboardBuilder()
    for k in keys[page * 6:(page + 1) * 6]:
        state = overrides.get(k, {}).get("state")
        mark = ("🔧 " if state == "maintenance"
                else ("🗑 " if state == "hidden"
                      else ("⭐ " if state == "custom" else "")))
        text = f"{mark}{MODELS[k].name}"
        callback_data = f"admin:models:tgl:{mode}:{cat_key}:{page}:{k}"
        button = _danger_button(text, callback_data) if mode == "x" else InlineKeyboardButton(
            text=text, callback_data=callback_data,
        )
        b.row(button)
    nav = []
    if page > 0:
        nav.append(InlineKeyboardButton(text="⬅️", callback_data=f"admin:models:list:{mode}:{cat_key}:{page - 1}"))
    if len(keys) > 6:
        nav.append(InlineKeyboardButton(text=f"{page + 1}/{page_count}", callback_data="admin:models:noop"))
    if (page + 1) * 6 < len(keys):
        nav.append(InlineKeyboardButton(text="➡️", callback_data=f"admin:models:list:{mode}:{cat_key}:{page + 1}"))
    if nav:
        b.row(*nav)
    b.row(InlineKeyboardButton(text="⬅️ К моделям", callback_data="admin:models"))
    return b


@router.callback_query(F.data == "admin:models:noop")
async def cb_admin_models_noop(cb: CallbackQuery) -> None:
    # Кнопка декоративная (заголовок списка), вреда нет — гвард стоит ради
    # правила «ни одного admin:* без проверки», чтобы аудит читался однозначно.
    if not _is_admin(cb.from_user.id):
        await cb.answer("⛔", show_alert=True)
        return
    await cb.answer()


@router.callback_query(F.data == "admin:models")
async def cb_admin_models(cb: CallbackQuery) -> None:
    if not _is_admin(cb.from_user.id):
        await cb.answer("⛔", show_alert=True)
        return
    await cb.answer()
    text = (
        "🧩 <b>Управление моделями</b>\n\n"
        "🔧 <b>Техработы</b> — модель помечается и выпадает из ротации "
        "(в пользовательском меню остаётся с уведомлением), повторный тап возвращает.\n"
        "🗑 <b>Скрыть</b> — модель полностью убирается из ротации, повторный тап возвращает.\n"
        "📝 <b>Описание</b> — добавить, изменить или убрать описание модели.\n"
        "➕ <b>Добавить</b> — своя модель (OpenRouter/Featherless id) в конец категории.\n"
        "⚠️ <b>Удалить</b> — удалить модель после отдельного подтверждения."
    )
    try:
        await cb.message.edit_text(text, reply_markup=_admin_models_menu_kb().as_markup())
    except Exception:
        pass


@router.callback_query(F.data.regexp(r"^admin:models:mode:[mhdx]$"))
async def cb_admin_models_mode(cb: CallbackQuery) -> None:
    if not _is_admin(cb.from_user.id):
        await cb.answer("⛔", show_alert=True)
        return
    await cb.answer()
    mode = cb.data.split(":")[3]
    b = InlineKeyboardBuilder()
    for ck, cat in CATEGORIES.items():
        b.row(InlineKeyboardButton(
            text=f"{cat.emoji} {cat.name}",
            callback_data=f"admin:models:list:{mode}:{ck}:0",
        ))
    b.row(InlineKeyboardButton(text="⬅️ К моделям", callback_data="admin:models"))
    titles = {
        "m": "🔧 Выберите категорию — тап по модели ставит или снимает техработы:",
        "h": "🗑 Выберите категорию — тап по модели скрывает или возвращает её:",
        "d": "📝 Выберите категорию, затем модель для изменения описания:",
        "x": "⚠️ Выберите категорию, затем модель для удаления:",
    }
    title = titles[mode]
    try:
        await cb.message.edit_text(title, reply_markup=b.as_markup())
    except Exception:
        pass


@router.callback_query(F.data.startswith("admin:models:list:"))
async def cb_admin_models_list(cb: CallbackQuery) -> None:
    if not _is_admin(cb.from_user.id):
        await cb.answer("⛔", show_alert=True)
        return
    await cb.answer()
    parts = cb.data.split(":")
    mode, cat_key, page = parts[3], parts[4], int(parts[5])
    cat = CATEGORIES.get(cat_key)
    if cat is None:
        await cb.answer("Категория не найдена", show_alert=True)
        return
    titles = {
        "m": f"🔧 Техработы — {cat.emoji} {cat.name}\nТап = включить или выключить:",
        "h": f"🗑 Скрытие — {cat.emoji} {cat.name}\nТап = скрыть или вернуть:",
        "d": f"📝 Описания — {cat.emoji} {cat.name}\nВыберите модель:",
        "x": f"⚠️ Удаление — {cat.emoji} {cat.name}\nВыберите модель:",
    }
    title = titles.get(mode, "Модели:")
    try:
        await cb.message.edit_text(
            title,
            reply_markup=_admin_models_list_kb(mode, cat_key, page).as_markup(),
        )
    except Exception:
        pass


@router.callback_query(F.data.startswith("admin:models:tgl:"))
async def cb_admin_models_toggle(cb: CallbackQuery) -> None:
    if not _is_admin(cb.from_user.id):
        await cb.answer("⛔", show_alert=True)
        return
    parts = cb.data.split(":")
    mode, cat_key, page = parts[3], parts[4], int(parts[5])
    key = ":".join(parts[6:])
    current = _model_overrides().get(key, {}).get("state")
    row = _model_overrides().get(key, {})
    if key not in MODELS:
        await cb.answer("Модель не найдена", show_alert=True)
        return
    if mode == "d":
        desc = _model_description(key)
        b = InlineKeyboardBuilder()
        b.row(InlineKeyboardButton(
            text="✏️ Добавить / изменить",
            callback_data=f"admin:models:desc:edit:{cat_key}:{page}:{key}",
        ))
        b.row(InlineKeyboardButton(
            text="🧹 Убрать описание",
            callback_data=f"admin:models:desc:clear:{cat_key}:{page}:{key}",
        ))
        b.row(InlineKeyboardButton(
            text="⬅️ Назад",
            callback_data=f"admin:models:list:d:{cat_key}:{page}",
        ))
        shown = html.quote(desc) if desc else "<i>Описание отсутствует</i>"
        await cb.answer()
        await cb.message.edit_text(
            f"📝 <b>{html.quote(MODELS[key].name)}</b>\n\n{shown}",
            reply_markup=b.as_markup(),
        )
        return
    if mode == "x":
        b = InlineKeyboardBuilder()
        b.row(_danger_button(
            "🗑 Да, удалить модель",
            f"admin:models:delete:yes:{cat_key}:{page}:{key}",
        ))
        b.row(InlineKeyboardButton(
            text="❌ Отмена",
            callback_data=f"admin:models:list:x:{cat_key}:{page}",
        ))
        await cb.answer()
        await cb.message.edit_text(
            f"⚠️ <b>Удаление модели</b>\n\n"
            f"Модель: <b>{html.quote(MODELS[key].name)}</b>\n\n"
            f"Вы действительно хотите удалить эту модель?",
            reply_markup=b.as_markup(),
        )
        return
    if mode == "m":
        _set_model_override(
            key, _normal_model_state(key, row) if current == "maintenance" else "maintenance"
        )
    else:
        _set_model_override(
            key, _normal_model_state(key, row) if current == "hidden" else "hidden"
        )
    await cb.answer("✅")
    try:
        await cb.message.edit_text(
            cb.message.text,
            reply_markup=_admin_models_list_kb(mode, cat_key, page).as_markup(),
        )
    except Exception:
        pass


@router.callback_query(F.data.startswith("admin:models:desc:edit:"))
async def cb_admin_model_description_edit(cb: CallbackQuery) -> None:
    if not _is_admin(cb.from_user.id):
        await cb.answer("⛔", show_alert=True)
        return
    parts = cb.data.split(":")
    cat_key, page = parts[4], int(parts[5])
    key = ":".join(parts[6:])
    if key not in MODELS:
        await cb.answer("Модель не найдена", show_alert=True)
        return
    admin_state[cb.from_user.id] = f"model_desc:{cat_key}:{page}:{key}"
    await cb.answer()
    await cb.message.answer(
        f"📝 Отправьте новое описание для модели "
        f"<b>{html.quote(MODELS[key].name)}</b>.\n\n"
        f"Чтобы отменить, отправьте /cancel."
    )


@router.callback_query(F.data.startswith("admin:models:desc:clear:"))
async def cb_admin_model_description_clear(cb: CallbackQuery) -> None:
    if not _is_admin(cb.from_user.id):
        await cb.answer("⛔", show_alert=True)
        return
    parts = cb.data.split(":")
    cat_key, page = parts[4], int(parts[5])
    key = ":".join(parts[6:])
    if key not in MODELS:
        await cb.answer("Модель не найдена", show_alert=True)
        return
    _set_model_description(key, "")
    await cb.answer("Описание убрано", show_alert=True)
    await cb.message.edit_text(
        f"📝 <b>{html.quote(MODELS[key].name)}</b>\n\n<i>Описание отсутствует</i>",
        reply_markup=_admin_model_description_kb(key, cat_key, page).as_markup(),
    )


def _admin_model_description_kb(key: str, cat_key: str, page: int) -> InlineKeyboardBuilder:
    b = InlineKeyboardBuilder()
    b.row(InlineKeyboardButton(
        text="✏️ Добавить / изменить",
        callback_data=f"admin:models:desc:edit:{cat_key}:{page}:{key}",
    ))
    b.row(InlineKeyboardButton(
        text="🧹 Убрать описание",
        callback_data=f"admin:models:desc:clear:{cat_key}:{page}:{key}",
    ))
    b.row(InlineKeyboardButton(
        text="⬅️ Назад",
        callback_data=f"admin:models:list:d:{cat_key}:{page}",
    ))
    return b


@router.callback_query(F.data.startswith("admin:models:delete:yes:"))
async def cb_admin_model_delete_confirm(cb: CallbackQuery) -> None:
    if not _is_admin(cb.from_user.id):
        await cb.answer("⛔", show_alert=True)
        return
    parts = cb.data.split(":")
    cat_key, page = parts[4], int(parts[5])
    key = ":".join(parts[6:])
    if key not in MODELS:
        await cb.answer("Модель уже удалена", show_alert=True)
        return
    name = MODELS[key].name
    _set_model_override(key, "deleted")
    if key not in BUILTIN_MODEL_KEYS:
        MODELS.pop(key, None)
    user_specific_model_keys = [uid for uid, selected in user_specific_model.items() if selected == key]
    for uid in user_specific_model_keys:
        user_specific_model.pop(uid, None)
    save_state()
    await cb.answer("Модель удалена", show_alert=True)
    await cb.message.edit_text(
        f"✅ Модель <b>{html.quote(name)}</b> удалена.",
        reply_markup=_admin_models_list_kb("x", cat_key, page).as_markup(),
    )


@router.callback_query(F.data == "admin:models:add")
async def cb_admin_models_add(cb: CallbackQuery) -> None:
    if not _is_admin(cb.from_user.id):
        await cb.answer("⛔", show_alert=True)
        return
    await cb.answer()
    admin_state[cb.from_user.id] = "waiting_model_add"
    await cb.message.answer(
        "➕ <b>Новая модель</b>\n\n"
        "Отправьте одним сообщением:\n"
        "<code>provider/точный-id | Название</code>\n\n"
        "Пример: <code>openai/gpt-5.5:free | GPT-5.5 Free</code>\n"
        "Для Featherless начните id с <code>featherless/...</code>\n\n"
        "Отмена: /admin",
    )


@router.callback_query(F.data.startswith("admin:models:addcat:"))
async def cb_admin_models_addcat(cb: CallbackQuery) -> None:
    if not _is_admin(cb.from_user.id):
        await cb.answer("⛔", show_alert=True)
        return
    await cb.answer()
    uid = cb.from_user.id
    state = admin_state.get(uid, "")
    if not state.startswith("model_add:"):
        await cb.answer("Сессия добавления истекла — начните заново", show_alert=True)
        return
    admin_state.pop(uid, None)
    try:
        pending = json.loads(state.split(":", 1)[1])
    except Exception:
        pending = {}
    cat_key = cb.data.split(":")[3]
    model_id = pending.get("id") or "unknown/model"
    name = pending.get("name") or model_id
    raw_key = "custom_" + re.sub(r"[^a-z0-9]+", "_", model_id.lower()).strip("_")[:20]
    key, n = raw_key, 1
    while key in MODELS or key in _model_overrides():
        n += 1
        key = f"{raw_key}_{n}"
    provider = "featherless" if model_id.lower().startswith("featherless/") else "openrouter"
    _set_model_override(key, "custom", {
        "id": model_id,
        "name": name,
        "emoji": "⭐",
        "desc": "",
        "knowledge": "2025",
        "provider": provider,
        "category": cat_key,
    })
    _sync_custom_models()
    cat_name = CATEGORIES[cat_key].name if cat_key in CATEGORIES else cat_key
    try:
        await cb.message.edit_text(
            f"✅ Модель <b>{html.quote(name)}</b> ({html.quote(model_id)}) добавлена в категорию «{cat_name}».\n"
            f"В ротации появится сразу. Удалить: 🧩 Модели → 🗑 → «{cat_name}» → тап по ⭐.",
            reply_markup=_admin_models_menu_kb().as_markup(),
        )
    except Exception:
        pass


@router.callback_query(F.data == "admin:models:cancel")
async def cb_admin_models_cancel(cb: CallbackQuery) -> None:
    if not _is_admin(cb.from_user.id):
        await cb.answer("⛔", show_alert=True)
        return
    admin_state.pop(cb.from_user.id, None)
    await cb_admin_menu(cb)


async def _admin_model_add_step(msg: Message) -> None:
    """Шаг ввода «id | Название» для новой модели (состояние waiting_model_add)."""
    uid = msg.from_user.id
    text = (msg.text or "").strip()
    if text.startswith("/"):
        admin_state.pop(uid, None)
        await msg.answer("Отменено.")
        return
    parts = [p.strip() for p in text.split("|", 1)]
    model_id = parts[0]
    name = parts[1] if len(parts) > 1 and parts[1] else model_id
    if "/" not in model_id or len(model_id) < 4:
        await msg.answer(
            "😕 Нужен формат <code>provider/точный-id | Название</code>, "
            "например <code>openai/gpt-5.5:free | GPT-5.5</code>. Ещё раз или /admin для отмены."
        )
        return
    admin_state[uid] = "model_add:" + json.dumps({"id": model_id, "name": name}, ensure_ascii=False)
    b = InlineKeyboardBuilder()
    for ck, cat in CATEGORIES.items():
        b.row(InlineKeyboardButton(
            text=f"{cat.emoji} {cat.name}",
            callback_data=f"admin:models:addcat:{ck}",
        ))
    b.row(InlineKeyboardButton(text="❌ Отмена", callback_data="admin:models:cancel"))
    await msg.answer(
        f"Модель: <b>{name}</b>\nID: <code>{model_id}</code>\n\nВ какую категорию добавить?",
        reply_markup=b.as_markup(),
    )


async def _admin_model_description_step(msg: Message) -> None:
    """Сохраняет новое описание модели, введённое в панели."""
    uid = msg.from_user.id
    state = admin_state.get(uid, "")
    parts = state.split(":", 3)
    if len(parts) != 4:
        admin_state.pop(uid, None)
        await msg.answer("Сессия изменения описания истекла.")
        return
    _, cat_key, page_raw, key = parts
    text = (msg.text or "").strip()
    if text.lower() in ("/cancel", "/admin"):
        admin_state.pop(uid, None)
        await msg.answer("Изменение описания отменено.")
        return
    if key not in MODELS:
        admin_state.pop(uid, None)
        await msg.answer("Модель не найдена.")
        return
    if not text:
        await msg.answer("Описание не может быть пустым. Отправьте текст или /cancel.")
        return
    description = text[:1500]
    _set_model_description(key, description)
    admin_state.pop(uid, None)
    page = int(page_raw) if page_raw.isdigit() else 0
    await msg.answer(
        f"✅ Описание модели <b>{html.quote(MODELS[key].name)}</b> сохранено.\n\n"
        f"{html.quote(description)}",
        reply_markup=_admin_model_description_kb(key, cat_key, page).as_markup(),
    )


if __name__ == "__main__":
    # Логи и в консоль, и в файл bot.log рядом с ботом (ротация: 3 файла по 2 МБ).
    # Файл переживает рестарты — причину ночного падения можно посмотреть утром
    # командой /getfile bot.log.
    from logging.handlers import RotatingFileHandler
    _log_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bot.log")
    _file_handler = RotatingFileHandler(
        _log_file, maxBytes=2 * 1024 * 1024, backupCount=3, encoding="utf-8")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout), _file_handler],
    )
    # Понятная ошибка вместо падения где-то в глубине, если .env не подхватился
    # (частая причина: файл не в рабочей папке процесса).
    _missing = [name for name, val in
                (("BOT_TOKEN", BOT_TOKEN), ("OPENROUTER_KEY", OPENROUTER_KEY))
                if not val]
    if _missing:
        sys.exit(
            f"⛔ В .env не заданы обязательные переменные: {', '.join(_missing)}.\n"
            f"   Рабочая папка: {os.getcwd()} — .env должен лежать здесь "
            f"или рядом с ботом."
        )
    if not ADMIN_IDS:
        print("⚠️ ADMIN_IDS не задан: админка, алерты об ошибках и автобэкапы работать не будут.")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("aiogram.client").setLevel(logging.WARNING)
    logging.getLogger("aiogram.dispatcher").setLevel(logging.WARNING)
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logging.info("👋 Стоп")
