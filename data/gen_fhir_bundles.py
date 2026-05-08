"""Generate FHIR R4 transaction bundles matching Synthea format.

Patient A (Chen Wei): 68M, HF(LVEF32%) + T2DM(HbA1c8.2%) + CKD4(eGFR28)
Patient B (Maria Santos): 55F, HFpEF(LVEF58%), no DM, no CKD
"""
import json
import uuid

IDENTIFIER_SYSTEM = "https://consilium.dev/patient"

def uid():
    return str(uuid.uuid4())

def _request(resource_type, resource_id):
    return {"method": "POST", "url": resource_type, "ifNoneExist": f"identifier={IDENTIFIER_SYSTEM}|{resource_id}"}

def patient_entry(name_given, name_family, birth_date, gender, race_code, race_display, ethnicity_code):
    pid = uid()
    return pid, {
        "fullUrl": f"urn:uuid:{pid}",
        "resource": {
            "resourceType": "Patient",
            "id": pid,
            "meta": {"profile": ["http://hl7.org/fhir/us/core/StructureDefinition/us-core-patient"]},
            "text": {"status": "generated", "div": f"<div xmlns=\"http://www.w3.org/1999/xhtml\">Generated for Consilium demo</div>"},
            "extension": [
                {"url": "http://hl7.org/fhir/us/core/StructureDefinition/us-core-race", "extension": [
                    {"url": "ombCategory", "valueCoding": {"system": "urn:oid:2.16.840.1.113883.6.238", "code": race_code, "display": race_display}},
                    {"url": "text", "valueString": race_display}
                ]},
                {"url": "http://hl7.org/fhir/us/core/StructureDefinition/us-core-ethnicity", "extension": [
                    {"url": "ombCategory", "valueCoding": {"system": "urn:oid:2.16.840.1.113883.6.238", "code": ethnicity_code, "display": "Not Hispanic or Latino"}},
                    {"url": "text", "valueString": "Not Hispanic or Latino"}
                ]},
                {"url": "http://hl7.org/fhir/us/core/StructureDefinition/us-core-birthsex", "valueCode": gender[0].upper()},
            ],
            "identifier": [{"system": IDENTIFIER_SYSTEM, "value": pid}],
            "name": [{"use": "official", "family": name_family, "given": [name_given]}],
            "gender": gender,
            "birthDate": birth_date,
            "address": [{"use": "home", "city": "Boston", "state": "MA", "country": "US"}],
        },
        "request": _request("Patient", pid),
    }

def condition_entry(patient_id, code_system, code, display, onset, clinical_status="active"):
    rid = uid()
    return {
        "fullUrl": f"urn:uuid:{rid}",
        "resource": {
            "resourceType": "Condition",
            "id": rid,
            "meta": {"profile": ["http://hl7.org/fhir/us/core/StructureDefinition/us-core-condition"]},
            "clinicalStatus": {"coding": [{"system": "http://terminology.hl7.org/CodeSystem/condition-clinical", "code": clinical_status}]},
            "verificationStatus": {"coding": [{"system": "http://terminology.hl7.org/CodeSystem/condition-ver-status", "code": "confirmed"}]},
            "category": [{"coding": [{"system": "http://terminology.hl7.org/CodeSystem/condition-category", "code": "problem-list-item"}]}],
            "code": {"coding": [{"system": code_system, "code": code, "display": display}], "text": display},
            "subject": {"reference": f"Patient/{patient_id}"},
            "onsetDateTime": onset,
        },
        "request": _request("Condition", rid),
    }

def observation_entry(patient_id, loinc_code, loinc_display, value, unit, date, category="laboratory"):
    rid = uid()
    return {
        "fullUrl": f"urn:uuid:{rid}",
        "resource": {
            "resourceType": "Observation",
            "id": rid,
            "status": "final",
            "category": [{"coding": [{"system": "http://terminology.hl7.org/CodeSystem/observation-category", "code": category}]}],
            "code": {"coding": [{"system": "http://loinc.org", "code": loinc_code, "display": loinc_display}], "text": loinc_display},
            "subject": {"reference": f"Patient/{patient_id}"},
            "effectiveDateTime": date,
            "valueQuantity": {"value": value, "unit": unit, "system": "http://unitsofmeasure.org", "code": unit},
        },
        "request": _request("Observation", rid),
    }

def medication_entry(patient_id, med_text, authored_on, status="active"):
    rid = uid()
    return {
        "fullUrl": f"urn:uuid:{rid}",
        "resource": {
            "resourceType": "MedicationRequest",
            "id": rid,
            "status": status,
            "intent": "order",
            "medicationCodeableConcept": {"text": med_text},
            "subject": {"reference": f"Patient/{patient_id}"},
            "authoredOn": authored_on,
            "dosageInstruction": [{"text": med_text}],
        },
        "request": _request("MedicationRequest", rid),
    }

def encounter_entry(patient_id, reason, date):
    rid = uid()
    return {
        "fullUrl": f"urn:uuid:{rid}",
        "resource": {
            "resourceType": "Encounter",
            "id": rid,
            "status": "finished",
            "class": {"system": "http://terminology.hl7.org/CodeSystem/v3-ActCode", "code": "AMB", "display": "ambulatory"},
            "type": [{"text": reason}],
            "subject": {"reference": f"Patient/{patient_id}"},
            "period": {"start": date, "end": date},
        },
        "request": _request("Encounter", rid),
    }


# ════════════════════════════════════════════════════════════════
# Patient A: Chen Wei — HF + T2DM + CKD4 (complex)
# ════════════════════════════════════════════════════════════════

pid_a, patient_a = patient_entry("Wei", "Chen", "1958-03-15", "male", "2106-3", "White", "2186-5")
entries_a = [patient_a]
entries_a.append(condition_entry(pid_a, "http://snomed.info/sct", "84114007", "Heart failure", "2022-06-10"))
entries_a.append(condition_entry(pid_a, "http://snomed.info/sct", "44054006", "Type 2 diabetes mellitus", "2015-03-20"))
entries_a.append(condition_entry(pid_a, "http://snomed.info/sct", "723188008", "Chronic kidney disease stage 4", "2023-01-15"))
entries_a.append(condition_entry(pid_a, "http://snomed.info/sct", "38341003", "Hypertension", "2010-08-01"))
entries_a.append(observation_entry(pid_a, "10230-1", "Left ventricular ejection fraction", 32, "%", "2026-01-15"))
entries_a.append(observation_entry(pid_a, "48642-3", "eGFR", 28, "mL/min/1.73m2", "2026-02-10"))
entries_a.append(observation_entry(pid_a, "4548-4", "Hemoglobin A1c", 8.2, "%", "2026-03-01"))
entries_a.append(observation_entry(pid_a, "30934-4", "BNP", 850, "pg/mL", "2026-03-01"))
entries_a.append(observation_entry(pid_a, "2160-0", "Creatinine", 2.1, "mg/dL", "2026-02-10"))
entries_a.append(observation_entry(pid_a, "2823-3", "Potassium", 5.1, "mEq/L", "2026-02-10"))
entries_a.append(medication_entry(pid_a, "Lisinopril 10mg daily", "2022-06-15"))
entries_a.append(medication_entry(pid_a, "Metformin 500mg twice daily", "2015-04-01"))
entries_a.append(medication_entry(pid_a, "Furosemide 40mg twice daily", "2022-06-15"))
entries_a.append(medication_entry(pid_a, "Aspirin 81mg daily", "2022-06-15"))
entries_a.append(medication_entry(pid_a, "Glipizide 5mg twice daily", "2018-09-01"))
entries_a.append(encounter_entry(pid_a, "Heart failure follow-up", "2026-03-01"))
entries_a.append(encounter_entry(pid_a, "Diabetes management", "2026-02-15"))

bundle_a = {"resourceType": "Bundle", "type": "transaction", "entry": entries_a}
with open("data/fhir_bundles/patient_chen_wei_hf_t2dm_ckd.json", "w") as f:
    json.dump(bundle_a, f, indent=2)
print(f"Patient A: Wei Chen — HF+T2DM+CKD4 — {len(entries_a)} entries")


# ════════════════════════════════════════════════════════════════
# Patient B: Maria Santos — HFpEF only (simple counter-example)
# ════════════════════════════════════════════════════════════════

pid_b, patient_b = patient_entry("Maria", "Santos", "1971-07-24", "female", "2106-3", "White", "2186-5")
entries_b = [patient_b]
entries_b.append(condition_entry(pid_b, "http://snomed.info/sct", "84114007", "Heart failure", "2024-08-10"))
entries_b.append(condition_entry(pid_b, "http://snomed.info/sct", "38341003", "Hypertension", "2018-02-01"))
entries_b.append(observation_entry(pid_b, "10230-1", "Left ventricular ejection fraction", 58, "%", "2026-01-20"))
entries_b.append(observation_entry(pid_b, "48642-3", "eGFR", 82, "mL/min/1.73m2", "2026-02-05"))
entries_b.append(observation_entry(pid_b, "4548-4", "Hemoglobin A1c", 5.4, "%", "2026-02-05"))
entries_b.append(observation_entry(pid_b, "30934-4", "BNP", 220, "pg/mL", "2026-02-05"))
entries_b.append(observation_entry(pid_b, "2160-0", "Creatinine", 0.9, "mg/dL", "2026-02-05"))
entries_b.append(observation_entry(pid_b, "2823-3", "Potassium", 4.2, "mEq/L", "2026-02-05"))
entries_b.append(medication_entry(pid_b, "Lisinopril 20mg daily", "2024-08-15"))
entries_b.append(medication_entry(pid_b, "Carvedilol 12.5mg twice daily", "2024-08-15"))
entries_b.append(encounter_entry(pid_b, "Heart failure follow-up", "2026-02-05"))

bundle_b = {"resourceType": "Bundle", "type": "transaction", "entry": entries_b}
with open("data/fhir_bundles/patient_maria_santos_hf_only.json", "w") as f:
    json.dump(bundle_b, f, indent=2)
print(f"Patient B: Maria Santos — HFpEF only — {len(entries_b)} entries")
