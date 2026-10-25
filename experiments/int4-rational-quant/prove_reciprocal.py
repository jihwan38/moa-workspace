#!/usr/bin/env python3
"""Exact integer reciprocal proof for the already specified rational A4 rule."""
from fractions import Fraction
import hashlib
import json
from pathlib import Path


def reference(mx, ma, delta):
    ratio = Fraction(7 * mx, ma * (1 << delta)) + Fraction(1, 2)
    return ratio.numerator // ratio.denominator


def reciprocal_code(mx, ma, delta, precision):
    delta = min(delta, 5)
    reciprocal = ((1 << precision) + ma - 1) // ma
    return (7 * mx * reciprocal + (1 << (precision - 1 + delta))) >> (
        precision + delta
    )


def main():
    cases = 0
    errors = {p: [] for p in range(19, 25)}
    observed_max = {p: 0 for p in errors}
    for delta in range(6):
        for ma in range(128, 256):
            for mx in range(128, 256):
                if delta == 0 and mx > ma:
                    continue
                expected = reference(mx, ma, delta)
                cases += 1
                for precision in errors:
                    actual = reciprocal_code(mx, ma, delta, precision)
                    if actual != expected:
                        errors[precision].append(
                            dict(mx=mx, ma=ma, delta=delta,
                                 expected=expected, actual=actual)
                        )
                    reciprocal = ((1 << precision) + ma - 1) // ma
                    intermediate = 7 * mx * reciprocal + (1 << (precision - 1 + delta))
                    observed_max[precision] = max(observed_max[precision], intermediate)
                    assert intermediate < (1 << 31)
    assert cases == 90176
    assert len(errors[19]) == 2
    assert all(not errors[p] for p in range(20, 25))

    # Every larger exponent difference is below 1/2 even at extremal mantissas.
    assert Fraction(7 * 255, 128 * 32) < Fraction(1, 2)
    for delta in range(5, 254):
        assert reference(255, 128, delta) == 0
        assert reciprocal_code(255, 128, delta, 20) == 0

    table = [((1 << 20) + ma - 1) // ma for ma in range(128, 256)]
    table_bytes = b''.join(value.to_bytes(4, 'little') for value in table)
    result = {
        "contract": "Existing exact rational finite-BF16 A4; FTZ subnormals; normalized A>=2^-112; sign restored after magnitude",
        "formula": "R=ceil(2^20/Ma); qmag=(7*Mx*R+(1<<(19+min(delta,5))))>>(20+min(delta,5))",
        "oracle": "fractions.Fraction exact quotient plus 1/2, then integer floor",
        "normalized_magnitude_cases_per_precision": cases,
        "precisions_checked": list(errors),
        "mismatches": {str(p): rows for p, rows in errors.items()},
        "observed_max_intermediate": observed_max,
        "signed_i32_upper_bound_p20": 31399936,
        "proof": {
            "reciprocal_error_numerator": "h=Ma*ceil(2^p/Ma)-2^p lies in[0,Ma-1]",
            "below_half_crossing_requirement": "2*n*h >= D*2^p for integer D=(2k+1)*Ma*2^delta-2*n>=1",
            "bound": "2*n*h <= 2*1785*254=906780 < 2^20; ties remain upward",
            "precision_19_fails": True,
            "larger_exponents": "delta>=5 remains zero even after clamping to5",
        },
        "reciprocal_table_u32_le_sha256": hashlib.sha256(table_bytes).hexdigest(),
        "reciprocal_table_ma128_through255": table,
        "scale_rule_unchanged": "RN32(A*RN32(1/7))",
        "device_lookup_implemented": False,
        "device_equivalence_verified": False,
        "performance_gain": None,
        "model_quality_verified": False,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    destination = Path(__file__).with_name('reciprocal-proof.json')
    destination.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({"result": "PASS integer proof only", "cases_per_precision": cases,
                      "p19_mismatches": len(errors[19]), "p20_through24_mismatches": 0,
                      "output": str(destination)}))


if __name__ == '__main__':
    main()
