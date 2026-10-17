from ci_ident.costs import INVALID_OPERATIONS, VALID_OPERATIONS, classify, extract_operation


def test_tables_disjoint_and_division_inadmissible():
    assert not set(VALID_OPERATIONS) & set(INVALID_OPERATIONS)
    for op in ("div", "rem", "fdiv"):
        assert classify(op) == "invalid"
    for label in ("%3 = sdiv i32 %1, %2", "%3 = udiv i32 %1, %2", "%3 = srem i32 %1, %2"):
        assert classify(extract_operation(label)) == "invalid"


def test_every_admissible_op_takes_at_most_one_cycle():
    assert all(0 < hw <= 1.0 and area > 0 for hw, area in VALID_OPERATIONS.values())


def test_extract_operation():
    assert extract_operation("%5 = add nsw i32 %3, %4") == "add"
    assert extract_operation("mul") == "mul"
    assert extract_operation("anything", op_attr="lshr") == "lshr"
    assert extract_operation("%2 = xor i32 %1, -1") == "not"
    assert extract_operation("  %2 = xor i32 %1, -1, !ddg.orig_bb !9", op_attr="xor") == "not"
    assert extract_operation("%2 = xor i1 %1, true") == "not"
    assert extract_operation("%2 = xor i32 %1, 1") == "xor"
    assert extract_operation("%2 = xor i32 %1, -12") == "xor"
    assert extract_operation("%2 = fneg float %1") == "neg"
    assert extract_operation("%r = call i32 @llvm.fshl.i32(i32 %a, i32 %b, i32 3)") == "rotl"
    assert extract_operation("%r = call float @llvm.fma.f32(float %a, float %b, float %c)") == "fma"
    assert extract_operation("%r = call i32 @foo(i32 %a)") == "call"
