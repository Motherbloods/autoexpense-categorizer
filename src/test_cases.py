import pickle
import sys
from pathlib import Path

# IMPORT CLASSES FROM train_model to make them available during unpickling
from train_model import (
    RuleAwareClassifier,
    KeywordFeatureExtractor,
    ContextAwareKeywordExtractor,
)

# ============================================================================
# COMPREHENSIVE TEST SUITE
# ============================================================================


def load_model(model_path="../models/expense_classifier.pkl"):
    """Load trained model"""
    with open(model_path, "rb") as f:
        return pickle.load(f)


def test_ambiguous_cases(model):
    """Test dengan kasus-kasus yang ambigu/kompleks"""
    print(f"\n{'='*70}")
    print("TEST 1: AMBIGUOUS & COMPLEX CASES")
    print(f"{'='*70}\n")

    test_cases = [
        # Ambiguitas: belanja vs langganan
        ("beli voucher game", "langganan"),  # game voucher = langganan
        ("beli voucher indomaret", "belanja"),  # voucher belanja = belanja
        ("top up mobile legend", "langganan"),  # game credit
        # Ambiguitas: tagihan vs transportasi
        ("bayar asuransi motor", "tagihan"),  # insurance = tagihan
        ("bayar bensin motor credit", "transportasi"),  # fuel = transportasi
        ("cicilan kendaraan", "tagihan"),  # installment = tagihan
        # Ambiguitas: hiburan vs rekreasi
        ("nonton konser di pantai", "hiburan"),  # konser = hiburan
        ("paket tour ke bali", "rekreasi"),  # tour = rekreasi
        ("camping di gunung", "rekreasi"),  # camping = rekreasi
        ("main playstation", "hiburan"),  # gaming = hiburan
        # Ambiguitas: belanja vs elektronik
        ("beli hp iphone terbaru", "elektronik"),  # phone = elektronik
        ("beli baju di toko online", "belanja"),  # clothing = belanja
        ("beli mouse dan keyboard", "elektronik"),  # accessories = elektronik
        # Ambiguitas: makanan vs rekreasi
        ("makan di restoran pinggir pantai", "makanan"),  # eating = makanan
        ("piknik dengan makan nasi kuning", "rekreasi"),  # piknik = rekreasi
        # Ambiguitas: perawatan vs elektronik
        ("facial di salon", "perawatan"),  # facial = perawatan
        ("beli alat facial skincare", "elektronik"),  # device = elektronik
        ("creambath di salon", "perawatan"),  # service = perawatan
    ]

    correct = 0
    total = len(test_cases)

    for activity, expected in test_cases:
        prediction = model.predict([activity])[0]
        is_correct = prediction == expected
        if is_correct:
            correct += 1

        status = "✓" if is_correct else "✗"
        print(f"{status} '{activity}'")
        print(f"   → Predicted: {prediction}, Expected: {expected}")
        if not is_correct:
            print(f"   ⚠️ MISMATCH!")
        print()

    print(f"{'='*70}")
    print(f"Accuracy: {correct}/{total} ({correct/total*100:.1f}%)")
    print(f"{'='*70}\n")

    return correct / total


def test_typos_and_variations(model):
    """Test dengan typo dan variasi penulisan"""
    print(f"\n{'='*70}")
    print("TEST 2: TYPOS & VARIATIONS")
    print(f"{'='*70}\n")

    test_cases = [
        # Typos
        ("beli nasi goreng", "makanan"),  # correct spelling
        ("beli nasi gorng", "makanan"),  # typo: gorng
        ("beli nas goreng", "makanan"),  # typo: nas
        ("beli nasi goring", "makanan"),  # typo: goring
        # Case sensitivity
        ("Beli Nasi Goreng", "makanan"),  # uppercase
        ("BELI NASI GORENG", "makanan"),  # all caps
        ("bELi NaSi gOrEng", "makanan"),  # mixed case
        # Extra spaces
        ("beli  nasi  goreng", "makanan"),  # double spaces
        ("  beli nasi goreng  ", "makanan"),  # leading/trailing spaces
        # Abbreviations & slang
        ("makan di resto", "makanan"),  # resto instead of restoran
        ("beli di toko", "belanja"),  # toko generic
        ("bayar PLN", "tagihan"),  # uppercase PLN
        ("top up pulsa", "langganan"),  # alternative phrasing
        ("isi bensin", "transportasi"),  # isi instead of bayar
        # Missing context
        ("bayar", "tagihan"),  # too generic
        ("beli", "belanja"),  # too generic
        ("makan", "makanan"),  # too generic
    ]

    correct = 0
    total = len(test_cases)

    for activity, expected in test_cases:
        prediction = model.predict([activity])[0]
        is_correct = prediction == expected
        if is_correct:
            correct += 1

        status = "✓" if is_correct else "✗"
        print(f"{status} '{activity}' → {prediction} (expected: {expected})")
        if not is_correct:
            print(f"   ⚠️ MISMATCH!")

    print(f"\n{'='*70}")
    print(f"Accuracy: {correct}/{total} ({correct/total*100:.1f}%)")
    print(f"{'='*70}\n")

    return correct / total


def test_edge_cases(model):
    """Test dengan edge case"""
    print(f"\n{'='*70}")
    print("TEST 3: EDGE CASES")
    print(f"{'='*70}\n")

    test_cases = [
        # Very short
        ("wifi", "tagihan"),  # just one word
        ("spotify", "langganan"),  # brand name
        # Very long
        (
            "pergi ke pantai dengan teman-teman untuk liburan musim panas dan makan seafood segar",
            "rekreasi",
        ),  # long sentence
        # Multiple categories mixed
        (
            "beli makanan dan minuman di restoran terus nonton bioskop",
            "makanan",
        ),  # first priority
        (
            "pergi liburan ke pantai tapi ada cicilan motor yang harus dibayar",
            "rekreasi",
        ),  # context matters
        # Negation/Exception
        ("tidak jadi beli", "belanja"),  # negation but still shopping context
        ("tidak bayar listrik", "tagihan"),  # still tagihan context
        # Numbers & symbols
        ("bayar rp 50000 bensin", "transportasi"),  # with amount
        ("cicilan 12x mobil", "tagihan"),  # with term
        ("beli hp seharga 10jt", "elektronik"),  # with price
        # Regional variations
        ("bayar tol jalan tol", "transportasi"),  # toll
        ("naik angkot", "transportasi"),  # public transport
        ("naik busway", "transportasi"),  # BRT
    ]

    correct = 0
    total = len(test_cases)

    for activity, expected in test_cases:
        prediction = model.predict([activity])[0]
        is_correct = prediction == expected
        if is_correct:
            correct += 1

        status = "✓" if is_correct else "✗"
        print(f"{status} '{activity}'")
        print(f"   → {prediction} (expected: {expected})")
        if not is_correct:
            print(f"   ⚠️ MISMATCH!")
        print()

    print(f"{'='*70}")
    print(f"Accuracy: {correct}/{total} ({correct/total*100:.1f}%)")
    print(f"{'='*70}\n")

    return correct / total


def test_weak_signals(model):
    """Test dengan signal yang lemah/minimal"""
    print(f"\n{'='*70}")
    print("TEST 4: WEAK SIGNALS (Minimal Keywords)")
    print(f"{'='*70}\n")

    test_cases = [
        # Medium-strength keywords only
        ("naik bis umum", "transportasi"),  # bis = medium keyword
        ("voucher game mobile", "langganan"),  # voucher = medium keyword
        ("kredit barang", "tagihan"),  # kredit = medium keyword
        ("musik karaoke", "hiburan"),  # musik = medium keyword
        ("alam terbuka", "rekreasi"),  # alam = medium keyword
        # Only strong keywords indirect match
        ("gofood pesanan", "makanan"),  # gofood is strong
        ("shopee checkout", "belanja"),  # shopee is strong
        ("spotify playlist", "langganan"),  # spotify is strong
        ("dokter check-up", "perawatan"),  # dokter is strong
        # Descriptive instead of direct
        ("kendaraan baru", "transportasi"),  # kendaraan = medium
        ("gadget terbaru", "elektronik"),  # gadget = strong
        ("layanan profesional", "jasa_profesional"),  # layanan = medium
    ]

    correct = 0
    total = len(test_cases)

    for activity, expected in test_cases:
        prediction = model.predict([activity])[0]
        is_correct = prediction == expected
        if is_correct:
            correct += 1

        status = "✓" if is_correct else "✗"
        print(f"{status} '{activity}' → {prediction} (expected: {expected})")
        if not is_correct:
            print(f"   ⚠️ MISMATCH!")

    print(f"\n{'='*70}")
    print(f"Accuracy: {correct}/{total} ({correct/total*100:.1f}%)")
    print(f"{'='*70}\n")

    return correct / total


def test_rule_engine_effectiveness(model):
    """Test khusus untuk rule engine"""
    print(f"\n{'='*70}")
    print("TEST 5: RULE ENGINE EFFECTIVENESS")
    print(f"{'='*70}\n")

    test_cases = [
        # Rule 1: beli pulsa = langganan (not belanja)
        ("beli pulsa telkomsel", "langganan"),
        ("isi kuota data", "langganan"),
        ("top up voucher game", "langganan"),
        # Rule 2: bayar cicilan = tagihan
        ("bayar cicilan motor", "tagihan"),
        ("bayar cicilan mobil", "tagihan"),
        ("angsuran kpr rumah", "tagihan"),
        # Rule 3: bayar utility = tagihan (not elektronik)
        ("bayar token listrik", "tagihan"),
        ("isi air pdam", "tagihan"),
        ("bayar gas elpiji", "tagihan"),
        ("top up wifi indihome", "tagihan"),
        # Rule 4: konsultan = jasa_profesional
        ("konsultasi pajak", "jasa_profesional"),
        ("bayar lawyer", "jasa_profesional"),
        ("jasa desainer grafis", "jasa_profesional"),
        # Rule 5: belanja tanpa payment keyword = belanja
        ("belanja bulanan di supermarket", "belanja"),
        ("belanja baju dan sepatu", "belanja"),
        # Rule 6: bayar parkir = transportasi (not tagihan)
        ("bayar parkir di mall", "transportasi"),
        ("parkir mobil", "transportasi"),
    ]

    correct = 0
    total = len(test_cases)

    for activity, expected in test_cases:
        prediction = model.predict([activity])[0]
        is_correct = prediction == expected
        if is_correct:
            correct += 1

        status = "✓" if is_correct else "✗"
        print(f"{status} '{activity}' → {prediction} (expected: {expected})")
        if not is_correct:
            print(f"   ⚠️ Rule tidak jalan!")

    print(f"\n{'='*70}")
    print(f"Accuracy: {correct}/{total} ({correct/total*100:.1f}%)")
    print(f"Rule Engine Effectiveness: {correct/total*100:.1f}%")
    print(f"{'='*70}\n")

    return correct / total


def main():
    """Run all tests"""
    print("\n" + "=" * 70)
    print("COMPREHENSIVE EXPENSE CLASSIFIER TEST SUITE")
    print("=" * 70)

    try:
        print("\nLoading model...")
        model = load_model()
        print("✓ Model loaded successfully!")
    except FileNotFoundError:
        print("✗ Model not found. Please train the model first.")
        return

    results = {}

    # Run all tests
    results["Ambiguous Cases"] = test_ambiguous_cases(model)
    results["Typos & Variations"] = test_typos_and_variations(model)
    results["Edge Cases"] = test_edge_cases(model)
    results["Weak Signals"] = test_weak_signals(model)
    results["Rule Engine"] = test_rule_engine_effectiveness(model)

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}\n")

    for test_name, accuracy in results.items():
        bar_length = int(accuracy * 30)
        bar = "█" * bar_length + "░" * (30 - bar_length)
        print(f"{test_name:.<25} {bar} {accuracy*100:>5.1f}%")

    overall = sum(results.values()) / len(results)
    print(f"\n{'Overall Accuracy':.>25} {overall*100:.1f}%")
    print(f"{'='*70}\n")

    # Analysis
    print("ANALYSIS:")
    worst_test = min(results, key=results.get)
    best_test = max(results, key=results.get)
    print(f"✓ Best performance: {best_test} ({results[best_test]*100:.1f}%)")
    print(f"✗ Needs improvement: {worst_test} ({results[worst_test]*100:.1f}%)")


if __name__ == "__main__":
    main()
