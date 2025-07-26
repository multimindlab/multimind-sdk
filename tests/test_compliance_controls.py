from multimind.compliance.iso import ISOControl

def test_iso_control_check_compliance():
    control = ISOControl(
        control_id="C1",
        standard="ISO9001",
        category="Quality",
        name="Test Control",
        description="A test control.",
        implementation_status="implemented"
    )
    assert control.check_compliance() is True
    control.implementation_status = "not_implemented"
    assert control.check_compliance() is False 