from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)

CRYPTO_BANNED_COUNTRIES = {"China", "North Korea"}
CRYPTO_RESTRICTED_COUNTRIES = {"India", "Nigeria"}


@dataclass
class Jurisdiction:
    country: str


class JurisdictionType(Enum):
    RED = "RED"
    YELLOW = "YELLOW"
    GREEN = "GREEN"


@dataclass
class UserContext:
    gps_location: str | None
    ip_address: str | None
    sim_country: str | None
    device_locale: str | None


class JurisdictionManager:
    """Detect and apply jurisdiction-specific economic rules."""

    def __init__(self) -> None:
        """Initialise a jurisdiction manager with default rules."""
        self.jurisdiction_rules: dict[str, dict[str, str]] = (
            self.load_jurisdiction_rules()
        )
        self.user_modes: dict[str, str] = {}
        self.disabled_features: dict[str, set] = {}

    def load_jurisdiction_rules(self) -> dict[str, dict[str, str]]:
        return {
            "China": {"status": "banned"},
            "North Korea": {"status": "banned"},
            "India": {"status": "restricted"},
            "Nigeria": {"status": "restricted"},
        }

    def resolve_jurisdiction(self, signals: dict[str, str | None]) -> Jurisdiction:
        for key in ["gps", "sim", "ip", "locale"]:
            country = signals.get(key)
            if country:
                logger.debug("Jurisdiction resolved via %s: %s", key, country)
                return Jurisdiction(country)
        logger.warning("Unable to resolve jurisdiction, defaulting to US")
        return Jurisdiction("United States")

    def detect_jurisdiction(self, user_context: UserContext) -> JurisdictionType:
        signals = {
            "gps": user_context.gps_location,
            "ip": user_context.ip_address,
            "sim": user_context.sim_country,
            "locale": user_context.device_locale,
        }
        jurisdiction = self.resolve_jurisdiction(signals)
        country = jurisdiction.country
        if country in CRYPTO_BANNED_COUNTRIES:
            return JurisdictionType.RED
        if country in CRYPTO_RESTRICTED_COUNTRIES:
            return JurisdictionType.YELLOW
        return JurisdictionType.GREEN

    def set_user_mode(self, user_id: str, mode: str) -> None:
        logger.info("Setting mode for %s to %s", user_id, mode)
        self.user_modes[user_id] = mode

    def disable_crypto_features(self, user_id: str) -> None:
        logger.info("Disabling crypto features for %s", user_id)
        self.disabled_features.setdefault(user_id, set()).update({"crypto"})

    def limit_crypto_features(self, user_id: str) -> None:
        logger.info("Limiting crypto features for %s", user_id)
        self.disabled_features.setdefault(user_id, set()).update({"defi"})

    def enable_mining(self, user_id: str) -> None:
        logger.info("Enabling mining for %s", user_id)

    def enable_defi(self, user_id: str) -> None:
        logger.info("Enabling DeFi for %s", user_id)

    def use_terminology(self, term: str) -> None:
        logger.debug("Using terminology: %s", term)

    def apply_jurisdiction_rules(
        self, user_id: str, jurisdiction: JurisdictionType
    ) -> None:
        if jurisdiction == JurisdictionType.RED:
            self.set_user_mode(user_id, "EDUCATION_ONLY")
            self.disable_crypto_features(user_id)
            self.use_terminology("achievement_points")
        elif jurisdiction == JurisdictionType.YELLOW:
            self.set_user_mode(user_id, "UTILITY_TOKEN")
            self.limit_crypto_features(user_id)
        else:
            self.set_user_mode(user_id, "FULL_CRYPTO")
            self.enable_mining(user_id)
            self.enable_defi(user_id)
