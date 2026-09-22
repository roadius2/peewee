"""Script and language detection. Pure Python, no weights."""
import pytest

from peewee_decide.lang import analyse, detect_script, guess_latin_language, is_english, state_text


@pytest.mark.parametrize("text, want", [
    ("The customer was charged twice and wants a refund.", "latin"),
    ("Le client a été facturé deux fois et demande un remboursement.", "latin"),
    ("ग्राहक से दो बार शुल्क लिया गया और वह धनवापसी चाहता है।", "devanagari"),
    ("お客様は二重に請求されたため返金を希望しています。", "kana"),
    ("客户被重复扣款要求退款", "han"),
    ("고객이 두 번 청구되어 환불을 원합니다", "hangul"),
    ("تم خصم المبلغ مرتين من العميل ويريد استرداد الأموال", "arabic"),
    ("வாடிக்கையாளரிடம் இருமுறை கட்டணம் வசூலிக்கப்பட்டது", "tamil"),
    ("С клиента дважды сняли деньги и он хочет возврат", "cyrillic"),
    ("ลูกค้าถูกเรียกเก็บเงินสองครั้งและต้องการเงินคืน", "thai"),
    ("Ο πελάτης χρεώθηκε δύο φορές και θέλει επιστροφή χρημάτων", "greek"),
    ("הלקוח חויב פעמיים ורוצה החזר כספי", "hebrew"),
    ("Հայերեն տեքստ", "armenian"),          # upstream issue #20 / PR #23
    ("", "unknown"),
    ("12345 6789", "unknown"),
])
def test_detect_script(text, want):
    assert detect_script(text) == want


@pytest.mark.parametrize("text, want", [
    ("Please refund the duplicate charge on invoice 4411 today.", True),
    ("refund me", True),
    ("ग्राहक से दो बार शुल्क लिया गया", False),
    ("お客様は二重に請求されました", False),
    ("С клиента дважды сняли деньги", False),
    ("Հայերեն տեքստ", False),
    ("Le client a été facturé deux fois et il demande un remboursement pour la "
     "facture qui a été payée le mois dernier avec la carte de crédit", False),
    ("Der Kunde wurde zweimal belastet und möchte eine Rückerstattung für die "
     "Rechnung die nicht korrekt ist und auch nicht bezahlt wurde", False),
])
def test_is_english(text, want):
    assert is_english(text) is want


@pytest.mark.parametrize("text, want", [
    ("The customer was charged twice and wants a refund for this invoice", "en"),
    ("Le client a ete facture deux fois et il demande un remboursement pour la facture", "fr"),
    ("Der Kunde wurde zweimal belastet und moechte eine Rueckerstattung fuer die Rechnung", "de"),
    ("El cliente fue cobrado dos veces y quiere que le devuelvan el dinero por la factura", "es"),
    ("refund", None),
    ("Please refund the duplicate charge on invoice 4411 today because "
     "we have been waiting for three days and nobody has replied to us", "en"),
])
def test_guess_latin_language(text, want):
    assert guess_latin_language(text) == want


# Short support-style messages in Latin-script languages must not reach the English checkpoint.
@pytest.mark.parametrize("text, lang", [
    ("Je veux annuler mon forfait", "fr"),
    ("Mon compte ne marche pas", "fr"),
    ("Bonjour, je n'arrive pas à me connecter à mon compte depuis hier", "fr"),
    ("Pouvez-vous me rappeler demain matin?", "fr"),
    ("Merci beaucoup", "fr"),
    ("Necesito cancelar mi cuenta hoy", "es"),
    ("Muchas gracias por su ayuda con la factura", "es"),
    ("Hola, no puedo entrar", "es"),
    ("Ich kann mich nicht einloggen", "de"),
    ("Danke schön", "de"),
    ("Mein Konto wurde zweimal belastet", "de"),
    ("Não consigo aceder à minha conta", "pt"),
    ("Non riesco ad accedere al mio account", "it"),
    ("Ik kan niet inloggen", "nl"),
])
def test_short_non_english_latin(text, lang):
    assert is_english(text) is False
    assert guess_latin_language(text) == lang


# ...and ordinary short English must not be pushed off the English checkpoint either, even
# when it contains words that double as function words elsewhere ("me", "no", "a", "la").
@pytest.mark.parametrize("text", [
    "refund me",
    "no problem",
    "call me back tomorrow",
    "is it done yet",
    "on my way",
    "the invoice is late",
    "a la carte menu please",
    "no, send me the invoice",
    "can you help me with my account",
    "I need to cancel my plan today",
    "ok",
    "thanks!",
])
def test_short_english_stays_english(text):
    assert is_english(text) is True


def test_external_detector_hook():
    from peewee_decide.lang import register_language_detector
    calls = []

    def det(text):
        calls.append(text)
        return "FR-ca"

    register_language_detector(det)
    try:
        assert guess_latin_language("anything at all here") == "fr"
        assert is_english("plain english text") is False
        register_language_detector(lambda t: (_ for _ in ()).throw(RuntimeError("boom")))
        assert is_english("plain english text here") is True          # a broken plug-in is ignored
    finally:
        register_language_detector(None)
    assert calls
    assert is_english("plain english text") is True


def test_state_text_flattening():
    assert "charged twice" in state_text({"body": "charged twice", "n": 3})
    assert "deep" in state_text({"a": {"b": ["deep"]}})
    assert "x" in state_text(["x", {"y": "z"}])
    assert state_text(None) == ""


def test_keys_do_not_drive_detection():
    assert analyse({"subject": "नमस्ते", "body": "ग्राहक से दो बार शुल्क लिया गया"})["is_english"] is False
