"""Script and language detection. Pure Python, no weights."""
import pytest

from laya.lang import analyse, detect_script, guess_latin_language, is_english, state_text


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


# Known gap, tracked for Phase 1: short non-English Latin text is still routed as English.
@pytest.mark.xfail(reason="Phase 1: short French/Spanish/German sentences still look English", strict=True)
@pytest.mark.parametrize("text", [
    "Je veux annuler mon forfait",
    "Mon compte ne marche pas",
    "Necesito cancelar mi cuenta hoy",
    "Ich kann mich nicht einloggen",
])
def test_short_non_english_latin(text):
    assert is_english(text) is False


def test_state_text_flattening():
    assert "charged twice" in state_text({"body": "charged twice", "n": 3})
    assert "deep" in state_text({"a": {"b": ["deep"]}})
    assert "x" in state_text(["x", {"y": "z"}])
    assert state_text(None) == ""


def test_keys_do_not_drive_detection():
    assert analyse({"subject": "नमस्ते", "body": "ग्राहक से दो बार शुल्क लिया गया"})["is_english"] is False
