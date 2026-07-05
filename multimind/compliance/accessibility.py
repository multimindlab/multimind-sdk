"""
Accessibility and anti-discrimination compliance implementation.
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from .governance import GovernanceConfig

logger = logging.getLogger("AccessibilityCompliance")


class AccessibilityCompliance(BaseModel):
    """Accessibility and anti-discrimination compliance manager."""

    config: GovernanceConfig
    assessment_records: Dict[str, Dict[str, Any]] = Field(default_factory=dict)

    @staticmethod
    def _safe_bool(value: Any) -> bool:
        """Strict boolean parser for evidence flags."""
        return value is True

    def _get_system_evidence(self, system_id: str) -> Dict[str, Any]:
        """Return evidence for a system from custom governance settings."""
        evidence_registry = self.config.custom_settings.get("accessibility_evidence", {})
        evidence = evidence_registry.get(system_id, {})
        if not evidence:
            logger.warning("No accessibility evidence found for system: %s", system_id)
        return evidence

    @staticmethod
    def _bool_control_status(
        evidence: Dict[str, Any], controls: List[str], category: str
    ) -> Dict[str, Any]:
        """Evaluate controls backed by boolean evidence flags."""
        missing_controls = [
            control
            for control in controls
            if not AccessibilityCompliance._safe_bool(evidence.get(control))
        ]
        status = "compliant" if not missing_controls else "non_compliant"
        return {
            "category": category,
            "controls": controls,
            "status": status,
            "missing_controls": missing_controls,
        }

    async def validate_wcag_compliance(
        self, assessment_id: str, system_id: str, version: str = "2.1"
    ) -> Dict[str, Any]:
        """Validate compliance with WCAG 2.1 guidelines."""
        evidence = self._get_system_evidence(system_id)
        if version != "2.1":
            logger.warning(
                "Unsupported WCAG version '%s'. Falling back to 2.1 controls.",
                version,
            )
        assessment = {
            "assessment_id": assessment_id,
            "framework": "WCAG",
            "version": version,
            "assessed_at": datetime.now(),
            "system_id": system_id,
            "requirements": [
                {
                    "principle": "perceivable",
                    "guidelines": [
                        {
                            "name": "text_alternatives",
                            "level": "A",
                            "controls": [
                                "alt_text",
                                "captions",
                                "audio_descriptions",
                                "sign_language",
                            ],
                        },
                        {
                            "name": "time_based_media",
                            "level": "A",
                            "controls": [
                                "captions",
                                "audio_descriptions",
                                "sign_language",
                                "media_alternatives",
                            ],
                        },
                        {
                            "name": "adaptable",
                            "level": "A",
                            "controls": [
                                "content_structure",
                                "presentation_control",
                                "sensory_characteristics",
                            ],
                        },
                        {
                            "name": "distinguishable",
                            "level": "AA",
                            "controls": [
                                "color_contrast",
                                "audio_control",
                                "text_resizing",
                                "images_of_text",
                            ],
                        },
                    ],
                },
                {
                    "principle": "operable",
                    "guidelines": [
                        {
                            "name": "keyboard_accessible",
                            "level": "A",
                            "controls": [
                                "keyboard_navigation",
                                "no_keyboard_trap",
                                "keyboard_shortcuts",
                                "focus_visible",
                            ],
                        },
                        {
                            "name": "enough_time",
                            "level": "A",
                            "controls": [
                                "timing_adjustable",
                                "pause_stop_hide",
                                "no_timing",
                                "interruptions",
                            ],
                        },
                        {
                            "name": "seizures",
                            "level": "A",
                            "controls": ["three_flashes", "three_flashes_below_threshold"],
                        },
                        {
                            "name": "navigable",
                            "level": "AA",
                            "controls": [
                                "bypass_blocks",
                                "page_titled",
                                "focus_order",
                                "link_purpose",
                            ],
                        },
                    ],
                },
                {
                    "principle": "understandable",
                    "guidelines": [
                        {
                            "name": "readable",
                            "level": "A",
                            "controls": [
                                "language_of_page",
                                "language_of_parts",
                                "unusual_words",
                                "abbreviations",
                            ],
                        },
                        {
                            "name": "predictable",
                            "level": "A",
                            "controls": [
                                "on_focus",
                                "on_input",
                                "consistent_navigation",
                                "consistent_identification",
                            ],
                        },
                        {
                            "name": "input_assistance",
                            "level": "AA",
                            "controls": [
                                "error_identification",
                                "labels_instructions",
                                "error_suggestion",
                                "error_prevention",
                            ],
                        },
                    ],
                },
                {
                    "principle": "robust",
                    "guidelines": [
                        {
                            "name": "compatible",
                            "level": "A",
                            "controls": ["parsing", "name_role_value", "status_messages"],
                        }
                    ],
                },
            ],
            "overall_status": "compliant",
        }

        missing_controls = []
        for principle in assessment["requirements"]:
            for guideline in principle["guidelines"]:
                guideline_missing = [
                    control
                    for control in guideline["controls"]
                    if not self._safe_bool(evidence.get(control))
                ]
                guideline["missing_controls"] = guideline_missing
                guideline["status"] = "compliant" if not guideline_missing else "non_compliant"
                missing_controls.extend(guideline_missing)

        assessment["overall_status"] = "compliant" if not missing_controls else "non_compliant"
        assessment["summary"] = {
            "checked_controls": len(
                set(
                    c
                    for p in assessment["requirements"]
                    for g in p["guidelines"]
                    for c in g["controls"]
                )
            ),
            "missing_controls": sorted(set(missing_controls)),
            "missing_count": len(set(missing_controls)),
        }

        self.assessment_records[assessment_id] = assessment
        return assessment

    async def validate_ada_compliance(
        self, assessment_id: str, system_id: str, title: str = "III"
    ) -> Dict[str, Any]:
        """Validate compliance with Americans with Disabilities Act."""
        evidence = self._get_system_evidence(system_id)
        requirements = [
            self._bool_control_status(
                evidence,
                [
                    "auxiliary_aids",
                    "qualified_interpreters",
                    "telecommunications",
                    "video_remote_interpreting",
                ],
                "effective_communication",
            ),
            self._bool_control_status(
                evidence,
                [
                    "policy_modifications",
                    "service_animals",
                    "mobility_devices",
                    "auxiliary_aids",
                ],
                "reasonable_modifications",
            ),
            self._bool_control_status(
                evidence,
                [
                    "physical_access",
                    "alternative_methods",
                    "service_animals",
                    "auxiliary_aids",
                ],
                "program_accessibility",
            ),
            self._bool_control_status(
                evidence,
                [
                    "website_accessibility",
                    "mobile_app_accessibility",
                    "electronic_documents",
                    "multimedia_accessibility",
                ],
                "digital_accessibility",
            ),
        ]
        overall_status = (
            "compliant"
            if all(r["status"] == "compliant" for r in requirements)
            else "non_compliant"
        )
        assessment = {
            "assessment_id": assessment_id,
            "framework": "ADA",
            "title": title,
            "assessed_at": datetime.now(),
            "system_id": system_id,
            "requirements": requirements,
            "overall_status": overall_status,
        }

        self.assessment_records[assessment_id] = assessment
        return assessment

    async def validate_equality_act(
        self, assessment_id: str, system_id: str, jurisdiction: str
    ) -> Dict[str, Any]:
        """Validate compliance with Equality Act requirements."""
        evidence = self._get_system_evidence(system_id)
        requirements = [
            self._bool_control_status(
                evidence,
                [
                    "age",
                    "disability",
                    "gender_reassignment",
                    "marriage_civil_partnership",
                    "pregnancy_maternity",
                    "race",
                    "religion_belief",
                    "sex",
                    "sexual_orientation",
                ],
                "protected_characteristics",
            ),
            self._bool_control_status(
                evidence,
                ["direct_discrimination", "indirect_discrimination", "harassment", "victimisation"],
                "prohibited_conduct",
            ),
            self._bool_control_status(
                evidence,
                ["physical_changes", "auxiliary_aids", "service_provision", "policy_changes"],
                "reasonable_adjustments",
            ),
            self._bool_control_status(
                evidence,
                ["encouragement", "training", "outreach", "monitoring"],
                "positive_action",
            ),
        ]
        overall_status = (
            "compliant"
            if all(r["status"] == "compliant" for r in requirements)
            else "non_compliant"
        )
        assessment = {
            "assessment_id": assessment_id,
            "framework": "EQUALITY_ACT",
            "jurisdiction": jurisdiction,
            "assessed_at": datetime.now(),
            "system_id": system_id,
            "requirements": requirements,
            "overall_status": overall_status,
        }

        self.assessment_records[assessment_id] = assessment
        return assessment

    async def get_assessment_history(
        self, assessment_id: Optional[str] = None, framework: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get assessment history."""
        if assessment_id:
            return [self.assessment_records.get(assessment_id, {})]

        if framework:
            return [
                record
                for record in self.assessment_records.values()
                if record.get("framework") == framework
            ]

        return list(self.assessment_records.values())
