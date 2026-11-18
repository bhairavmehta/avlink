import pandas as pd

from avlink import normalize as nz
from avlink.sources import avherald


def test_norm_reg_us_prefix_and_separators():
    assert nz.norm_reg("5798T", "US") == "N5798T"
    assert nz.norm_reg("n5798t") == "N5798T"
    assert nz.norm_reg("D-AIZA") == "DAIZA"
    assert nz.norm_reg("9V-SKA") == "9VSKA"
    assert nz.norm_reg("") is None
    assert nz.norm_reg("UNK") is None


def test_dms_and_decimal():
    assert abs(nz.parse_coord("362543N", True) - 36.4286) < 1e-3
    assert abs(nz.parse_coord("0942135W", False) + 94.3597) < 1e-3
    assert nz.parse_coord("47.45", True) == 47.45
    assert nz.parse_coord("999", True) is None


def test_type_family():
    assert nz.type_family("BOEING", "737-9") == "B737MAX"
    assert nz.type_family("Boeing 737-9 MAX") == "B737MAX"
    assert nz.type_family("B39M") == "B737MAX"
    assert nz.type_family("BOEING", "737-800") == "B737NG"
    assert nz.type_family("Airbus A321-231") == "A320F"
    assert nz.type_family("CESSNA", "172S") == "C172"
    assert nz.type_family("Cessna 172S Skyhawk SP") == "C172"
    assert nz.type_family("EMBRAER", "ERJ 170-200 LR") == "E-JET"
    assert nz.type_family("E75L") == "E-JET"


def test_flight_and_operator():
    assert nz.norm_flight_no("UA-1854") == "UA1854"
    assert nz.norm_flight_no("UAL01854") == "UA1854"
    assert nz.operator_icao("United Airlines") == "UAL"
    assert nz.operator_icao("PSA Airlines Inc") == "JIA"


def test_ntsb_refs():
    txt = "see ERA21LA043 and https://data.ntsb.gov/Docket?ProjectID=103412"
    assert nz.extract_ntsb_numbers(txt) == ["ERA21LA043"]
    assert nz.extract_docket_ids(txt) == ["103412"]


def test_utc():
    ts = nz.to_utc(pd.Timestamp("2026-10-25"), "1435", "CDT")
    assert ts.hour == 19 and ts.minute == 35


def test_avherald_headline():
    h = avherald.parse_headline("Incident: United B39M at Houston on Oct 25th 2026, unsafe gear")
    assert h["op"] == "United" and h["type"] == "B39M" and h["place"] == "Houston"
    assert h["date"] == pd.Timestamp("2026-10-25")
    p = avherald.parse_first_paragraph(
        "A United Boeing 737-9 MAX, registration N27474 performing flight UA-1854 from George Town "
        "(Cayman Islands) to Houston Intercontinental,TX (USA) with 83 passengers and 6 crew, was ...")
    assert p["reg"] == "N27474" and p["flight"] == "UA-1854" and p["occupants"] == 89
