"""
Infrastructure Classifier

Analyzes and classifies infrastructure components for diversity validation.
Detects ASN, TEE vendor, power grid region, and network topology characteristics.
"""

import asyncio
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
import hashlib
import ipaddress
import json
import logging

try:
    import geoip2.database
    import geoip2.errors
    GEOIP2_AVAILABLE = True
except ImportError:
    GEOIP2_AVAILABLE = False
    geoip2 = None

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False
    requests = None


class TEEVendor(Enum):
    """TEE hardware vendors"""
    AMD_SEV_SNP = "amd-sev-snp"
    INTEL_TDX = "intel-tdx"
    ARM_TRUSTZONE = "arm-trustzone"
    UNKNOWN = "unknown"


class PowerRegion(Enum):
    """Power grid regions for diversity"""
    NERC_RFC = "nerc-rfc"       # ReliabilityFirst
    NERC_SERC = "nerc-serc"     # SERC Reliability
    NERC_TRE = "nerc-tre"       # Texas RE
    NERC_WECC = "nerc-wecc"     # Western Electricity
    NERC_MRO = "nerc-mro"       # Midwest Reliability
    NERC_NPCC = "nerc-npcc"     # Northeast Power
    INTERNATIONAL = "international"
    UNKNOWN = "unknown"


@dataclass
class InfrastructureProfile:
    """Complete infrastructure classification"""
    device_id: str
    asn: int | None
    asn_name: str | None
    tee_vendor: TEEVendor
    tee_version: str | None
    power_region: PowerRegion
    country_code: str
    region: str
    city: str
    network_topology: str
    attestation_hash: str | None
    classification_time: datetime
    confidence_score: float  # 0.0-1.0


class InfrastructureClassifier:
    """Classifies infrastructure components for diversity validation"""

    def __init__(self, geoip_db_path: str | None = None):
        self.geoip_db_path = geoip_db_path or "data/GeoLite2-City.mmdb"
        self.asn_cache: dict[str, tuple[int, str]] = {}
        self.cache_ttl = timedelta(hours=24)
        self._last_cache_clean = datetime.utcnow()

        # Power grid region mappings (simplified)
        self.power_region_mapping = {
            'US': {
                'Connecticut': PowerRegion.NERC_NPCC,
                'Maine': PowerRegion.NERC_NPCC,
                'Massachusetts': PowerRegion.NERC_NPCC,
                'New Hampshire': PowerRegion.NERC_NPCC,
                'New York': PowerRegion.NERC_NPCC,
                'Rhode Island': PowerRegion.NERC_NPCC,
                'Vermont': PowerRegion.NERC_NPCC,
                'Delaware': PowerRegion.NERC_RFC,
                'Maryland': PowerRegion.NERC_RFC,
                'New Jersey': PowerRegion.NERC_RFC,
                'Ohio': PowerRegion.NERC_RFC,
                'Pennsylvania': PowerRegion.NERC_RFC,
                'Virginia': PowerRegion.NERC_RFC,
                'West Virginia': PowerRegion.NERC_RFC,
                'Texas': PowerRegion.NERC_TRE,
                'Alabama': PowerRegion.NERC_SERC,
                'Florida': PowerRegion.NERC_SERC,
                'Georgia': PowerRegion.NERC_SERC,
                'Kentucky': PowerRegion.NERC_SERC,
                'Mississippi': PowerRegion.NERC_SERC,
                'North Carolina': PowerRegion.NERC_SERC,
                'South Carolina': PowerRegion.NERC_SERC,
                'Tennessee': PowerRegion.NERC_SERC,
                'Arizona': PowerRegion.NERC_WECC,
                'California': PowerRegion.NERC_WECC,
                'Colorado': PowerRegion.NERC_WECC,
                'Idaho': PowerRegion.NERC_WECC,
                'Montana': PowerRegion.NERC_WECC,
                'Nevada': PowerRegion.NERC_WECC,
                'New Mexico': PowerRegion.NERC_WECC,
                'Oregon': PowerRegion.NERC_WECC,
                'Utah': PowerRegion.NERC_WECC,
                'Washington': PowerRegion.NERC_WECC,
                'Wyoming': PowerRegion.NERC_WECC,
                'Iowa': PowerRegion.NERC_MRO,
                'Minnesota': PowerRegion.NERC_MRO,
                'Nebraska': PowerRegion.NERC_MRO,
                'North Dakota': PowerRegion.NERC_MRO,
                'South Dakota': PowerRegion.NERC_MRO,
            }
        }

    async def classify_device(self,
                            device_id: str,
                            ip_address: str,
                            attestation_data: dict | None = None,
                            network_info: dict | None = None) -> InfrastructureProfile:
        """
        Classify a device's infrastructure characteristics

        Args:
            device_id: Unique device identifier
            ip_address: Device IP address for ASN/geo lookup
            attestation_data: TEE attestation data
            network_info: Additional network topology information

        Returns:
            Complete infrastructure profile
        """
        # Clean cache periodically
        await self._clean_cache()

        # Classify in parallel
        tasks = [
            self._classify_asn(ip_address),
            self._classify_geo_location(ip_address),
            self._classify_tee_vendor(attestation_data),
            self._classify_network_topology(ip_address, network_info)
        ]

        asn_info, geo_info, tee_info, network_topology = await asyncio.gather(*tasks)

        # Determine power region
        power_region = self._determine_power_region(
            geo_info.get('country_code', ''),
            geo_info.get('region', '')
        )

        # Generate attestation hash
        attestation_hash = None
        if attestation_data:
            attestation_hash = hashlib.sha256(
                json.dumps(attestation_data, sort_keys=True).encode()
            ).hexdigest()[:16]

        # Calculate confidence score
        confidence_score = self._calculate_confidence_score(
            asn_info, geo_info, tee_info, attestation_data
        )

        return InfrastructureProfile(
            device_id=device_id,
            asn=asn_info.get('asn'),
            asn_name=asn_info.get('name'),
            tee_vendor=tee_info['vendor'],
            tee_version=tee_info.get('version'),
            power_region=power_region,
            country_code=geo_info.get('country_code', ''),
            region=geo_info.get('region', ''),
            city=geo_info.get('city', ''),
            network_topology=network_topology,
            attestation_hash=attestation_hash,
            classification_time=datetime.utcnow(),
            confidence_score=confidence_score
        )

    async def _classify_asn(self, ip_address: str) -> dict:
        """Classify ASN information"""
        try:
            # Check cache first
            if ip_address in self.asn_cache:
                cache_time, cached_data = self.asn_cache[ip_address]
                if datetime.utcnow() - cache_time < self.cache_ttl:
                    return cached_data[1]

            if not REQUESTS_AVAILABLE:
                return {'asn': None, 'name': None}

            # Query ASN information (using ipinfo.io as example)
            response = requests.get(
                f"https://ipinfo.io/{ip_address}/json",
                timeout=5
            )

            if response.status_code == 200:
                data = response.json()
                org = data.get('org', '')

                # Extract ASN number
                asn = None
                if org.startswith('AS'):
                    try:
                        asn = int(org.split()[0][2:])
                    except (ValueError, IndexError):
                        pass

                asn_info = {
                    'asn': asn,
                    'name': org.replace(f'AS{asn} ', '') if asn else org
                }

                # Cache result
                self.asn_cache[ip_address] = (datetime.utcnow(), asn_info)

                return asn_info

        except Exception as e:
            print(f"ASN classification error for {ip_address}: {e}")

        return {'asn': None, 'name': None}

    async def _classify_geo_location(self, ip_address: str) -> dict:
        """Classify geographic location"""
        try:
            if GEOIP2_AVAILABLE:
                with geoip2.database.Reader(self.geoip_db_path) as reader:
                    response = reader.city(ip_address)

                    return {
                        'country_code': response.country.iso_code,
                        'country_name': response.country.name,
                        'region': response.subdivisions.most_specific.name,
                        'city': response.city.name,
                        'latitude': float(response.location.latitude or 0),
                        'longitude': float(response.location.longitude or 0)
                    }

        except Exception as e:
            # Fallback to IP-based geolocation service
            if REQUESTS_AVAILABLE:
                try:
                    response = requests.get(
                        f"https://ipapi.co/{ip_address}/json/",
                        timeout=5
                    )

                    if response.status_code == 200:
                        data = response.json()
                        return {
                            'country_code': data.get('country_code', ''),
                            'country_name': data.get('country_name', ''),
                            'region': data.get('region', ''),
                            'city': data.get('city', ''),
                            'latitude': data.get('latitude', 0),
                            'longitude': data.get('longitude', 0)
                        }
                except Exception:
                    logging.exception("Failed to parse geo location data from IP service response")

        return {
            'country_code': '',
            'country_name': '',
            'region': '',
            'city': '',
            'latitude': 0,
            'longitude': 0
        }

    async def _classify_tee_vendor(self, attestation_data: dict | None) -> dict:
        """Classify TEE vendor from attestation data"""
        if not attestation_data:
            return {'vendor': TEEVendor.UNKNOWN, 'version': None}

        # AMD SEV-SNP detection
        if any(key in attestation_data for key in ['sev_snp', 'amd_sev', 'snp']):
            return {
                'vendor': TEEVendor.AMD_SEV_SNP,
                'version': attestation_data.get('snp_version')
            }

        # Intel TDX detection
        if any(key in attestation_data for key in ['tdx', 'intel_tdx', 'sgx']):
            return {
                'vendor': TEEVendor.INTEL_TDX,
                'version': attestation_data.get('tdx_version')
            }

        # ARM TrustZone detection
        if any(key in attestation_data for key in ['trustzone', 'arm_tz', 'secure_world']):
            return {
                'vendor': TEEVendor.ARM_TRUSTZONE,
                'version': attestation_data.get('tz_version')
            }

        # Check platform info
        platform = attestation_data.get('platform', '').lower()
        if 'amd' in platform or 'sev' in platform:
            return {'vendor': TEEVendor.AMD_SEV_SNP, 'version': None}
        elif 'intel' in platform or 'tdx' in platform:
            return {'vendor': TEEVendor.INTEL_TDX, 'version': None}
        elif 'arm' in platform:
            return {'vendor': TEEVendor.ARM_TRUSTZONE, 'version': None}

        return {'vendor': TEEVendor.UNKNOWN, 'version': None}

    async def _classify_network_topology(self,
                                       ip_address: str,
                                       network_info: dict | None) -> str:
        """Classify network topology characteristics"""
        topology_indicators = []

        # IP address analysis
        try:
            ip = ipaddress.ip_address(ip_address)
            if ip.is_private:
                topology_indicators.append("private")
            elif ip.is_global:
                topology_indicators.append("public")

            if ip.version == 6:
                topology_indicators.append("ipv6")
            else:
                topology_indicators.append("ipv4")

        except ValueError:
            topology_indicators.append("invalid_ip")

        # Network info analysis
        if network_info:
            if network_info.get('datacenter'):
                topology_indicators.append("datacenter")
            if network_info.get('mobile'):
                topology_indicators.append("mobile")
            if network_info.get('residential'):
                topology_indicators.append("residential")
            if network_info.get('vpn'):
                topology_indicators.append("vpn")
            if network_info.get('satellite'):
                topology_indicators.append("satellite")

        return "_".join(topology_indicators) if topology_indicators else "unknown"

    def _determine_power_region(self, country_code: str, region: str) -> PowerRegion:
        """Determine power grid region"""
        if country_code == 'US' and region in self.power_region_mapping['US']:
            return self.power_region_mapping['US'][region]
        elif country_code and country_code != 'US':
            return PowerRegion.INTERNATIONAL
        else:
            return PowerRegion.UNKNOWN

    def _calculate_confidence_score(self,
                                   asn_info: dict,
                                   geo_info: dict,
                                   tee_info: dict,
                                   attestation_data: dict | None) -> float:
        """Calculate classification confidence score"""
        score = 0.0

        # ASN information (25%)
        if asn_info.get('asn'):
            score += 0.25
        elif asn_info.get('name'):
            score += 0.15

        # Geographic information (25%)
        if geo_info.get('country_code'):
            score += 0.15
            if geo_info.get('region'):
                score += 0.1

        # TEE vendor detection (30%)
        if tee_info['vendor'] != TEEVendor.UNKNOWN:
            score += 0.2
            if tee_info.get('version'):
                score += 0.1

        # Attestation data quality (20%)
        if attestation_data:
            score += 0.1
            if len(attestation_data) > 3:  # Rich attestation data
                score += 0.1

        return min(score, 1.0)

    async def _clean_cache(self):
        """Clean expired cache entries"""
        now = datetime.utcnow()

        if now - self._last_cache_clean > timedelta(hours=1):
            expired_keys = [
                key for key, (timestamp, _) in self.asn_cache.items()
                if now - timestamp > self.cache_ttl
            ]

            for key in expired_keys:
                del self.asn_cache[key]

            self._last_cache_clean = now

    def get_diversity_metrics(self, profiles: list[InfrastructureProfile]) -> dict:
        """Calculate diversity metrics for a set of profiles"""
        if not profiles:
            return {
                'asn_diversity': 0,
                'tee_vendor_diversity': 0,
                'power_region_diversity': 0,
                'geographic_diversity': 0,
                'total_diversity_score': 0.0
            }

        # Count unique values
        unique_asns = len(set(p.asn for p in profiles if p.asn))
        unique_tee_vendors = len(set(p.tee_vendor for p in profiles))
        unique_power_regions = len(set(p.power_region for p in profiles))
        unique_countries = len(set(p.country_code for p in profiles if p.country_code))

        # Calculate diversity scores
        total_devices = len(profiles)
        asn_diversity = unique_asns / total_devices if total_devices > 0 else 0
        tee_diversity = unique_tee_vendors / total_devices if total_devices > 0 else 0
        power_diversity = unique_power_regions / total_devices if total_devices > 0 else 0
        geo_diversity = unique_countries / total_devices if total_devices > 0 else 0

        # Weighted total diversity score
        total_diversity_score = (
            asn_diversity * 0.3 +
            tee_diversity * 0.3 +
            power_diversity * 0.25 +
            geo_diversity * 0.15
        )

        return {
            'asn_diversity': asn_diversity,
            'tee_vendor_diversity': tee_diversity,
            'power_region_diversity': power_diversity,
            'geographic_diversity': geo_diversity,
            'total_diversity_score': total_diversity_score,
            'unique_asns': unique_asns,
            'unique_tee_vendors': unique_tee_vendors,
            'unique_power_regions': unique_power_regions,
            'unique_countries': unique_countries,
            'total_devices': total_devices
        }
