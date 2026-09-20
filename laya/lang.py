"""Dependency-free language/script detection used to route between Laya checkpoints.

Routing only needs one decision: *is this English Latin text, or is it something the English
checkpoint cannot read?* Benchmarks on MASSIVE (14 languages) showed the English checkpoint
collapsing to near-random on non-Latin scripts (Hindi 0.100, Korean 0.103, Swahili 0.103,
Tamil 0.113 at 20 options, where random is 0.050), while holding up far better on Latin-script
languages (French 0.487, Spanish 0.480). So the signal that matters most is *script*, and the
secondary signal is whether Latin text is English.

Script detection is exact. The Latin-script language guess is a stopword/diacritic heuristic and
is explicitly best-effort: pass an explicit model or `lang=` when you already know the language.
"""
import re
from typing import Dict, List, Optional, Union

# Unicode blocks that the English (ModernBERT-large, 50k English BPE) checkpoint cannot read.
_SCRIPT_RANGES = [
    ("greek", ((0x0370, 0x03FF), (0x1F00, 0x1FFF))),
    ("cyrillic", ((0x0400, 0x052F), (0x2DE0, 0x2DFF), (0xA640, 0xA69F))),
    ("armenian", ((0x0530, 0x058F), (0xFB13, 0xFB17))),
    ("hebrew", ((0x0590, 0x05FF),)),
    ("arabic", ((0x0600, 0x06FF), (0x0750, 0x077F), (0x08A0, 0x08FF), (0xFB50, 0xFDFF), (0xFE70, 0xFEFF))),
    ("devanagari", ((0x0900, 0x097F), (0xA8E0, 0xA8FF))),
    ("bengali", ((0x0980, 0x09FF),)),
    ("gurmukhi", ((0x0A00, 0x0A7F),)),
    ("gujarati", ((0x0A80, 0x0AFF),)),
    ("oriya", ((0x0B00, 0x0B7F),)),
    ("tamil", ((0x0B80, 0x0BFF),)),
    ("telugu", ((0x0C00, 0x0C7F),)),
    ("kannada", ((0x0C80, 0x0CFF),)),
    ("malayalam", ((0x0D00, 0x0D7F),)),
    ("sinhala", ((0x0D80, 0x0DFF),)),
    ("thai", ((0x0E00, 0x0E7F),)),
    ("lao", ((0x0E80, 0x0EFF),)),
    ("tibetan", ((0x0F00, 0x0FFF),)),
    ("myanmar", ((0x1000, 0x109F),)),
    ("georgian", ((0x10A0, 0x10FF),)),
    ("ethiopic", ((0x1200, 0x137F),)),
    ("khmer", ((0x1780, 0x17FF),)),
    ("hangul", ((0x1100, 0x11FF), (0x3130, 0x318F), (0xAC00, 0xD7AF))),
    ("kana", ((0x3040, 0x309F), (0x30A0, 0x30FF), (0x31F0, 0x31FF))),
    ("han", ((0x3400, 0x4DBF), (0x4E00, 0x9FFF), (0xF900, 0xFAFF))),
]

# Function words. Latin-script languages overlap heavily (de/la/le/un/e/que), so each hit is
# weighted and a margin is required before calling something non-English. The lists lean
# towards the short, first-person words that dominate support messages ("je veux", "mon compte",
# "ich kann", "necesito"), because those are the inputs the old lists missed.
_STOP = {
    "en": {"the", "and", "is", "are", "was", "were", "to", "of", "in", "for", "with", "that",
           "this", "it", "you", "have", "has", "not", "but", "on", "at", "be", "as", "from",
           "will", "can", "would", "there", "their", "what", "which", "please", "we", "i",
           "my", "me", "your", "our", "am", "do", "does", "did", "cannot", "can't", "don't",
           "since", "yesterday", "today", "tomorrow", "thanks", "hello", "hi", "need", "want",
           "help", "if", "when", "how", "why", "still", "again", "just", "also"},
    "fr": {"le", "la", "les", "des", "une", "est", "pour", "dans", "que", "qui", "avec", "sur",
           "pas", "plus", "nous", "vous", "être", "cette", "mais", "sont", "ont", "aux", "ce",
           "je", "tu", "il", "elle", "on", "mon", "ma", "mes", "ton", "ta", "tes", "son", "sa",
           "ses", "ne", "du", "au", "et", "ou", "où", "à", "y", "en", "votre", "notre", "vos",
           "nos", "ces", "donc", "car", "si", "quand", "comment", "pourquoi", "depuis", "hier",
           "demain", "aujourd", "très", "bien", "toujours", "jamais", "encore", "déjà", "veux",
           "peux", "faut", "peut", "suis", "ai", "avez", "avons", "aussi", "chez", "sans",
           "j", "n", "qu", "c", "d", "l", "m", "s", "bonjour", "bonsoir", "merci", "svp",
           "oui", "non", "me", "moi", "toi", "lui", "compte", "marche", "rappeler", "fois"},
    "de": {"der", "die", "das", "und", "ist", "ein", "eine", "den", "dem", "nicht", "mit", "für",
           "auf", "von", "zu", "sich", "auch", "werden", "wurde", "haben", "sind", "oder", "aber",
           "ich", "du", "er", "sie", "es", "wir", "ihr", "mich", "mir", "dich", "dir", "uns",
           "mein", "meine", "meinem", "meinen", "dein", "kein", "keine", "kann", "könnte", "muss",
           "will", "möchte", "habe", "hat", "bin", "bist", "war", "wird", "ja", "nein", "bitte",
           "danke", "hallo", "seit", "gestern", "heute", "morgen", "noch", "schon", "sehr",
           "immer", "wieder", "hier", "dort", "wenn", "weil", "dass", "ob", "nur", "mal",
           "etwas", "nichts", "alles", "konto", "einloggen", "im", "am", "beim", "zum", "zur"},
    "es": {"el", "los", "las", "que", "por", "con", "para", "una", "es", "se", "del", "como",
           "pero", "son", "está", "este", "esta", "todo", "más", "muy", "hay", "sus",
           "yo", "tú", "él", "ella", "mi", "mis", "tu", "tus", "su", "nos", "les", "lo",
           "sí", "hoy", "ayer", "mañana", "ahora", "necesito", "quiero", "puedo", "tengo",
           "están", "fue", "ser", "gracias", "hola", "cuenta", "favor", "porque", "cuando",
           "cómo", "qué", "bien", "también", "ya", "todavía", "usted", "ustedes", "nuestro",
           "nuestra", "cancelar", "ayuda", "un", "y", "o", "al", "en", "de"},
    "pt": {"os", "as", "que", "em", "um", "uma", "para", "com", "não", "é", "se", "do", "da",
           "dos", "das", "mas", "são", "está", "este", "esta", "muito", "pelo", "pela",
           "eu", "você", "ele", "ela", "nós", "meu", "minha", "meus", "minhas", "seu", "sua",
           "hoje", "ontem", "amanhã", "agora", "preciso", "quero", "posso", "tenho", "obrigado",
           "obrigada", "olá", "conta", "por", "favor", "porque", "quando", "como", "também",
           "ainda", "já", "ao", "à", "no", "na", "nos", "nas", "de", "o", "a", "e"},
    "it": {"il", "lo", "gli", "che", "di", "per", "con", "non", "è", "si", "del", "della", "sono",
           "questo", "questa", "anche", "come", "più", "nella", "alla",
           "io", "tu", "lui", "lei", "noi", "voi", "mio", "mia", "miei", "mie", "tuo", "suo",
           "oggi", "ieri", "domani", "adesso", "ho", "hai", "ha", "posso", "voglio", "devo",
           "grazie", "ciao", "prego", "salve", "conto", "perché", "quando", "ancora", "già",
           "un", "una", "e", "o", "al", "dal", "nel", "sul", "le", "la", "i"},
    "nl": {"het", "een", "van", "is", "op", "te", "dat", "niet", "met", "voor", "zijn", "aan",
           "door", "maar", "ook", "worden", "deze", "naar", "wordt",
           "ik", "je", "jij", "u", "hij", "zij", "we", "wij", "mijn", "jouw", "uw", "kan",
           "moet", "wil", "heb", "heeft", "ben", "bent", "was", "vandaag", "gisteren", "morgen",
           "nog", "al", "graag", "dank", "bedankt", "alstublieft", "hallo", "rekening", "sinds",
           "en", "de", "of", "als", "dan", "er", "geen", "wel"},
}
# Words that are unambiguously not English. One of these with no English function word
# present is enough to route away from the English checkpoint, even in a two-word message.
_STRONG = {
    "fr": {"bonjour", "bonsoir", "merci", "svp", "veux", "peux", "pouvez", "voulez", "rappeler"},
    "de": {"danke", "bitte", "ich", "nicht", "kann", "möchte", "einloggen"},
    "es": {"gracias", "hola", "necesito", "quiero", "usted", "cancelar", "mañana"},
    "pt": {"obrigado", "obrigada", "olá", "preciso", "não", "você"},
    "it": {"grazie", "ciao", "prego", "voglio", "perché"},
    "nl": {"bedankt", "alstublieft", "graag", "ik", "niet"},
}
# Only words that are common in exactly one language count at full weight. Words shared by
# several languages ("la", "en", "de", "no"...) count as half a hit each.
_SHARED = {w for lg, sw in _STOP.items() for w in sw
           if sum(1 for lg2, sw2 in _STOP.items() if w in sw2) > 1}
_NON_EN_DIACRITICS = set("àâäãáåçéèêëíìîïñóòôöõøúùûüýÿßæœđłşţğıåäö")
SHORT_INPUT_WORDS = 12

# Optional external detector, e.g. lingua or fastText, registered with
# `register_language_detector(fn)`. It receives the flattened state text and returns an
# ISO-639-1 code (or None to fall back to the heuristic below).
_EXTERNAL_DETECTOR = None


def register_language_detector(fn) -> None:
    """Route Latin-script language guessing through `fn(text) -> code | None`; None clears it."""
    global _EXTERNAL_DETECTOR
    _EXTERNAL_DETECTOR = fn


_WORD = re.compile(r"[^\W\d_]+", re.UNICODE)


def _iter_text(state: Union[str, dict, list, None], _depth: int = 0) -> List[str]:
    """Collect the string leaves of a state (str / dict / list), so detection sees real content."""
    if _depth > 6 or state is None:
        return []
    if isinstance(state, str):
        return [state]
    if isinstance(state, dict):
        out = []
        for v in state.values():
            out.extend(_iter_text(v, _depth + 1))
        return out
    if isinstance(state, (list, tuple)):
        out = []
        for v in state:
            out.extend(_iter_text(v, _depth + 1))
        return out
    return []


def state_text(state: Union[str, dict, list, None], max_chars: int = 4000) -> str:
    """Flatten a state into the text used for detection (keys are ignored: they are usually English)."""
    return " ".join(_iter_text(state))[:max_chars]


def detect_script(text: str) -> str:
    """Dominant script of `text`: 'latin', 'han', 'devanagari', ... or 'unknown' if there are no letters."""
    counts: Dict[str, int] = {}
    latin = 0
    for ch in text:
        if not ch.isalpha():
            continue
        cp = ord(ch)
        if cp < 0x0250 or 0x1E00 <= cp <= 0x1EFF:      # Latin + Latin Extended Additional
            latin += 1
            continue
        for name, ranges in _SCRIPT_RANGES:
            if any(lo <= cp <= hi for lo, hi in ranges):
                counts[name] = counts.get(name, 0) + 1
                break
    counts["latin"] = latin
    total = sum(counts.values())
    if total == 0:
        return "unknown"
    return max(counts.items(), key=lambda kv: kv[1])[0]


def script_profile(text: str) -> Dict[str, float]:
    """Fraction of alphabetic characters belonging to each detected script."""
    counts: Dict[str, int] = {"latin": 0}
    for ch in text:
        if not ch.isalpha():
            continue
        cp = ord(ch)
        if cp < 0x0250 or 0x1E00 <= cp <= 0x1EFF:
            counts["latin"] += 1
            continue
        for name, ranges in _SCRIPT_RANGES:
            if any(lo <= cp <= hi for lo, hi in ranges):
                counts[name] = counts.get(name, 0) + 1
                break
    total = sum(counts.values())
    if not total:
        return {}
    return {k: v / total for k, v in counts.items() if v}


def _stop_scores(words: List[str]) -> Dict[str, float]:
    scores = {}
    for lg, sw in _STOP.items():
        scores[lg] = sum((0.5 if w in _SHARED else 1.0) for w in words if w in sw)
    return scores


def guess_latin_language(text: str) -> Optional[str]:
    """Best-effort language code for Latin-script text, or None when undecided.

    Scores function-word hits per language (shared words count half) and applies rules that
    favour catching non-English: the multilingual checkpoint is only a little weaker on
    English, while the English checkpoint collapses on everything else, so a wrong "not
    English" costs far less than a wrong "English".

      1. an external detector, if one is registered
      2. a strong marker ("merci", "ich", "gracias"...) with no English function words
      3. a non-English language with at least two hits and no English function words
      4. a non-English language beating English by two hits
      5. non-English diacritics plus at least one hit for a language that ties or beats English
      6. otherwise English if any English function word was seen, else undecided
    """
    if _EXTERNAL_DETECTOR is not None:
        try:
            code = _EXTERNAL_DETECTOR(text)
        except Exception:      # a broken plug-in must not take routing down with it
            code = None
        if code:
            return str(code).lower().split("-")[0]
    words = [w.lower() for w in _WORD.findall(text)]
    if not words:
        return None
    scores = _stop_scores(words)
    en = scores.get("en", 0.0)
    best_lg, best = max(((lg, sc) for lg, sc in scores.items() if lg != "en"),
                        key=lambda kv: kv[1], default=(None, 0.0))
    lowered = text.lower()
    diac = sum(1 for ch in lowered if ch in _NON_EN_DIACRITICS)
    diac_rate = diac / max(1, len(lowered))

    if en == 0:
        for lg, strong in _STRONG.items():
            if any(w in strong for w in words):
                return lg
        if best_lg and best >= 2:
            return best_lg
    if best_lg and best >= max(2, en + 2):
        return best_lg
    if best_lg and best >= 1 and best >= en and (diac_rate >= 0.04 or (diac >= 1 and len(words) <= SHORT_INPUT_WORDS)):
        return best_lg
    if en > 0 and en >= best:
        return "en"
    return None


def analyse(state: Union[str, dict, list, None]) -> Dict[str, object]:
    """Full detection result for a state.

    Returns `script`, `script_profile`, `language` (best effort, may be None),
    `is_english` and `non_latin_fraction`.
    """
    text = state_text(state)
    prof = script_profile(text)
    script = detect_script(text)
    non_latin = round(1.0 - prof.get("latin", 0.0), 4) if prof else 0.0
    if script == "unknown":
        return {"script": "unknown", "script_profile": prof, "language": None,
                "is_english": True, "non_latin_fraction": 0.0}
    if script != "latin":
        return {"script": script, "script_profile": prof, "language": None,
                "is_english": False, "non_latin_fraction": non_latin}
    lang = guess_latin_language(text)
    return {"script": "latin", "script_profile": prof, "language": lang,
            "is_english": lang in (None, "en"), "non_latin_fraction": non_latin}


def is_english(state: Union[str, dict, list, None]) -> bool:
    """True when the English checkpoint can be expected to read this state."""
    return bool(analyse(state)["is_english"])
