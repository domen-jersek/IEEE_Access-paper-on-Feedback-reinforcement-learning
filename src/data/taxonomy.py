"""
Build a documented 15-class intent taxonomy from label_auto.
Outputs: taxonomy.csv (mapping table) and adds 'intent_class' + 'intent_class_id' columns.
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)

INTENT_CLASS_MAP: dict[str, str] = {
    "to request admin rights on the computer": "admin_rights",
    "to request Kimble permission changes": "admin_rights",
    "to request an employee password reset": "password_reset",
    "to enable Client-to-Site VPN tunnel requests": "vpn_access",
    "to report a problem with network access": "vpn_access",
    "for a new Site-2-Site tunnel with a partner company or customer": "vpn_access",
    "for a new Site-to-Site tunnel with a partner company or customer": "vpn_access",
    "to request a software or a licence on my Windows device": "software_license",
    "to request a software or a license on my Windows device": "software_license",
    "if you need a Jira license": "software_license",
    "if you need a Confluence license": "software_license",
    "to support you about local software / programs": "software_support",
    "to be exclude of product updates": "software_support",
    "to report a problem with your devices": "hardware_support",
    "for additional hardware": "hardware_support",
    "On-boarding": "onboarding",
    "to start the IT On-boarding of a new internal employee": "onboarding",
    "to start the IT On-boarding of a new Contractor": "onboarding",
    "Off-boarding": "offboarding",
    "to start the IT Off-Boarding of an employee/consultant": "offboarding",
    "to request an Absence ticket": "absence",
    "to request office access": "office_access",
    "to customize a project": "project_tools",
    "to request a new project in JIRA": "project_tools",
    "to invite people outside of our Organization for collaboration": "collaboration",
    "to request sender to be excluded from html attachment blocking": "email_support",
    "to manage new mailbox requests": "email_support",
    "to change an existing e-mail distribution group": "email_support",
    "to request a new email distribution group": "email_support",
    "if you have problems with Outlook365 Calendar": "email_support",
    "to connect a newly created Supplier in Synertrade with an existing one in SAP": "enterprise_systems",
    "to report an issue with Engagement/Sales Opportunity.": "enterprise_systems",
    "for engagement / sales opportunity request": "enterprise_systems",
    "to import data to CRM": "enterprise_systems",
    "for a new SAP user": "enterprise_systems",
    "to manage the approval threshold requests": "enterprise_systems",
    "if you cannot access to JIRA although I have a license assigned": "access_problem",
    "to provide access to a server": "vpn_access",
    "for a new network access from a server or service inside of AZK network": "vpn_access",
    "for a new network access from a server or service inside of EQN network": "vpn_access",
    "for a new network access from a server or service inside of GFT network": "vpn_access",
    "for a new network access from a server or service inside of XYZ network": "vpn_access",
    "to ask for support regarding a timesheet": "hr_tools",
    "for support regarding a travel request / journey": "hr_tools",
    "for support regarding a resource": "hr_tools",
    "for o evaluate new Office365 / Microsoft Teams functionallity": "collaboration",
    "to renew a cloud subscription": "enterprise_systems",
    "to renew a cloud subcription": "enterprise_systems",
    "Italian": "other",
    "if you need additional support": "other",
    "other": "other",
}

_INTENT_ID: dict[str, int] = {}


def _class_name_to_id(class_name: str) -> int:
    if class_name not in _INTENT_ID:
        _INTENT_ID[class_name] = len(_INTENT_ID)
    return _INTENT_ID[class_name]


def build_taxonomy(df: pd.DataFrame, output_csv: Path) -> pd.DataFrame:
    log.info("Building intent taxonomy from label_auto (118 labels)")

    unique_labels = sorted(df["label_auto"].dropna().unique())
    mapping_rows = []
    unmapped = []

    for label in unique_labels:
        intent = INTENT_CLASS_MAP.get(label, None)
        if intent is None:
            unmapped.append(label)
            intent = "other"
        mapping_rows.append({
            "label_auto": label,
            "intent_class": intent,
            "intent_class_id": _class_name_to_id(intent),
        })

    mapping_df = pd.DataFrame(mapping_rows)
    mapping_df.to_csv(output_csv, index=False)
    log.info("Taxonomy written to %s (%d labels, %d classes, %d unmapped)", 
             output_csv, len(mapping_rows), len(_INTENT_ID), len(unmapped))

    if unmapped:
        log.warning("Unmapped labels -> 'other': %s", unmapped[:20])

    intent_map = dict(zip(mapping_df["label_auto"], mapping_df["intent_class"]))
    df["intent_class"] = df["label_auto"].map(intent_map).fillna("other")
    df["intent_class_id"] = df["intent_class"].apply(_class_name_to_id)

    log.info("Class distribution:\n%s", df["intent_class"].value_counts().to_string())
    return df


def get_intent_class_names() -> dict[int, str]:
    return {v: k for k, v in _INTENT_ID.items()}