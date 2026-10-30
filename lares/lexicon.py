"""Keyword lexicon used for (i) proxy emergency labels and (ii) selecting the domain-adaptation set.

The labels are *automatically generated proxy labels*, not human-validated ground truth
(see Sec. 3.3 of the paper). When several categories match, a fixed priority order decides.
The term lists below are a starting point written for this repository; if the paper's
experiments used a different lexicon, replace LEXICON with that exact list and commit it.
"""
from __future__ import annotations
import re

CATEGORIES = ["Medical", "Fire", "Disaster", "Police", "Neutral"]
PRIORITY = ["Medical", "Fire", "Disaster", "Police"]   # tie-break order (Sec. 3.3)

LEXICON = {
    "Medical": ["krankenhaus", "arzt", "ärzte", "notarzt", "rettungswagen", "krankenwagen", "verletzt",
                "verletzte", "verletzung", "herzinfarkt", "bewusstlos", "blutung", "sanitäter",
                "hospital", "doctor", "ambulance", "paramedic", "injured", "injury", "unconscious",
                "heart attack", "bleeding", "medical"],
    "Fire": ["feuer", "brand", "brennt", "flammen", "rauch", "feuerwehr", "löschen", "brandmelder",
             "fire", "burning", "flames", "smoke", "firefighter", "extinguish", "blaze"],
    "Disaster": ["erdbeben", "überschwemmung", "hochwasser", "sturm", "tsunami", "evakuierung",
                 "evakuieren", "katastrophe", "lawine", "waldbrand", "orkan", "tornado",
                 "earthquake", "flood", "storm", "evacuation", "evacuate", "disaster", "avalanche",
                 "hurricane", "wildfire", "landslide"],
    "Police": ["polizei", "polizist", "verbrechen", "einbruch", "überfall", "verhaftet", "festnahme",
               "raub", "bombe", "geisel", "schüsse", "schuss", "terror",
               "police", "crime", "burglary", "robbery", "arrested", "bomb", "hostage", "gunshot",
               "terror"],
}
# "Waldbrand"/"wildfire" appear under both Fire and Disaster on purpose: Fire wins by priority.
LEXICON["Fire"] += ["waldbrand", "wildfire"]

_PATTERNS = {c: re.compile(r"(?<!\w)(" + "|".join(re.escape(w) for w in ws) + r")", re.IGNORECASE)
             for c, ws in LEXICON.items()}


def matched_categories(text: str) -> list[str]:
    return [c for c in PRIORITY if _PATTERNS[c].search(text)]


def label_sentence(text: str) -> str:
    """Return the proxy category: first match in PRIORITY order, else 'Neutral'."""
    m = matched_categories(text)
    return m[0] if m else "Neutral"
