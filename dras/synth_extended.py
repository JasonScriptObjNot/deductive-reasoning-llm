"""
Extended training chains — approximately 20 full-context examples per pattern.

These are unannotated (plain-string steps) so they all go through full-context
masking in build_chains(): premises = seeds + all prior steps, target = next step.

Pattern counts:
  modus_tollens         — 10 chains,  20 rows
  hypothetical_syllogism —  6 chains,  20 rows
  disjunctive_syllogism  —  6 chains,  22 rows  (consequence rules always present)
  contrapositive_chain   —  5 chains,  22 rows
  mixed_negation         —  4 chains,  18 rows
  long_forward_chain     —  4 chains,  20 rows
                                    ──────────
  Total                             122 new full-context training rows
"""

from __future__ import annotations

EXT_CHAINS: list[dict] = [

    # ── MODUS TOLLENS EXTENDED ───────────────────────────────────────────────

    {
        "id": "mt_ext_01", "pattern": "modus_tollens",
        "seeds": [
            "All scholarship recipients have maintained a GPA above 3.5.",
            "Leon has not maintained a GPA above 3.5.",
        ],
        "goal": "Leon is not a scholarship recipient.",
        "steps": [
            "If someone has not maintained a GPA above 3.5, they are not a scholarship recipient.",
            "Leon is not a scholarship recipient.",
        ],
    },
    {
        "id": "mt_ext_02", "pattern": "modus_tollens",
        "seeds": [
            "Any building that meets fire code has an operational sprinkler system.",
            "The warehouse does not have an operational sprinkler system.",
        ],
        "goal": "The warehouse does not meet fire code.",
        "steps": [
            "If a building does not have an operational sprinkler system, it does not meet fire code.",
            "The warehouse does not meet fire code.",
        ],
    },
    {
        "id": "mt_ext_03", "pattern": "modus_tollens",
        "seeds": [
            "All club members have paid the annual subscription fee.",
            "Henderson has not paid the annual subscription fee.",
        ],
        "goal": "Henderson is not a club member.",
        "steps": [
            "If someone has not paid the annual subscription fee, they are not a club member.",
            "Henderson is not a club member.",
        ],
    },
    {
        "id": "mt_ext_04", "pattern": "modus_tollens",
        "seeds": [
            "All licensed architects have passed the professional licensing examination.",
            "Ms. Tanaka has not passed the professional licensing examination.",
        ],
        "goal": "Ms. Tanaka is not a licensed architect.",
        "steps": [
            "If someone has not passed the professional licensing examination, they are not a licensed architect.",
            "Ms. Tanaka is not a licensed architect.",
        ],
    },
    {
        "id": "mt_ext_05", "pattern": "modus_tollens",
        "seeds": [
            "Any system that is GDPR compliant has completed a data privacy audit.",
            "The CRM platform has not completed a data privacy audit.",
        ],
        "goal": "The CRM platform is not GDPR compliant.",
        "steps": [
            "If a system has not completed a data privacy audit, it is not GDPR compliant.",
            "The CRM platform is not GDPR compliant.",
        ],
    },
    {
        "id": "mt_ext_06", "pattern": "modus_tollens",
        "seeds": [
            "All publicly listed companies have filed quarterly financial reports.",
            "Axiom Industries has not filed quarterly financial reports.",
        ],
        "goal": "Axiom Industries is not a publicly listed company.",
        "steps": [
            "If a company has not filed quarterly financial reports, it is not a publicly listed company.",
            "Axiom Industries is not a publicly listed company.",
        ],
    },
    {
        "id": "mt_ext_07", "pattern": "modus_tollens",
        "seeds": [
            "All board-certified cardiologists have completed a cardiology fellowship.",
            "Dr. Osei has not completed a cardiology fellowship.",
        ],
        "goal": "Dr. Osei is not a board-certified cardiologist.",
        "steps": [
            "If someone has not completed a cardiology fellowship, they are not a board-certified cardiologist.",
            "Dr. Osei is not a board-certified cardiologist.",
        ],
    },
    {
        "id": "mt_ext_08", "pattern": "modus_tollens",
        "seeds": [
            "Any bridge approved for public use has passed structural load testing.",
            "The Eastfield bridge has not passed structural load testing.",
        ],
        "goal": "The Eastfield bridge is not approved for public use.",
        "steps": [
            "If a bridge has not passed structural load testing, it is not approved for public use.",
            "The Eastfield bridge is not approved for public use.",
        ],
    },
    {
        "id": "mt_ext_09", "pattern": "modus_tollens",
        "seeds": [
            "All airworthy aircraft have completed scheduled maintenance.",
            "Flight 409 has not completed scheduled maintenance.",
        ],
        "goal": "Flight 409 is not airworthy.",
        "steps": [
            "If an aircraft has not completed scheduled maintenance, it is not airworthy.",
            "Flight 409 is not airworthy.",
        ],
    },
    {
        "id": "mt_ext_10", "pattern": "modus_tollens",
        "seeds": [
            "All restaurants certified for food safety have passed a health inspection.",
            "The Harbour Grill has not passed a health inspection.",
        ],
        "goal": "The Harbour Grill is not certified for food safety.",
        "steps": [
            "If a restaurant has not passed a health inspection, it is not certified for food safety.",
            "The Harbour Grill is not certified for food safety.",
        ],
    },

    # ── HYPOTHETICAL SYLLOGISM EXTENDED ─────────────────────────────────────

    {
        "id": "hs_ext_01", "pattern": "hypothetical_syllogism",
        "seeds": [
            "If the glacier melts, sea levels rise.",
            "If sea levels rise, coastal flooding increases.",
            "If coastal flooding increases, coastal infrastructure requires reinforcement.",
            "If coastal infrastructure requires reinforcement, emergency funds are allocated.",
            "The glacier has melted.",
        ],
        "goal": "Emergency funds have been allocated.",
        "steps": [
            "Sea levels have risen.",
            "Coastal flooding has increased.",
            "Coastal infrastructure requires reinforcement.",
            "Emergency funds have been allocated.",
        ],
    },
    {
        "id": "hs_ext_02", "pattern": "hypothetical_syllogism",
        "seeds": [
            "If a contract is breached, the injured party may sue for damages.",
            "If a party sues for damages, a court date is scheduled.",
            "If a court date is scheduled, both parties must appear with legal counsel.",
            "The contract has been breached.",
        ],
        "goal": "Both parties must appear with legal counsel.",
        "steps": [
            "The injured party may sue for damages.",
            "A court date has been scheduled.",
            "Both parties must appear with legal counsel.",
        ],
    },
    {
        "id": "hs_ext_03", "pattern": "hypothetical_syllogism",
        "seeds": [
            "If the build pipeline fails, a bug report is generated.",
            "If a bug report is generated, the engineering team is notified.",
            "If the engineering team is notified, a remediation ticket is opened.",
            "If a remediation ticket is opened, a developer is assigned to fix the issue.",
            "The build pipeline has failed.",
        ],
        "goal": "A developer has been assigned to fix the issue.",
        "steps": [
            "A bug report has been generated.",
            "The engineering team has been notified.",
            "A remediation ticket has been opened.",
            "A developer has been assigned to fix the issue.",
        ],
    },
    {
        "id": "hs_ext_04", "pattern": "hypothetical_syllogism",
        "seeds": [
            "If a student misses the final exam, they receive an incomplete grade.",
            "If a student receives an incomplete grade, they must retake the course.",
            "If a student must retake the course, they are removed from the graduation list.",
            "Carlos missed the final exam.",
        ],
        "goal": "Carlos has been removed from the graduation list.",
        "steps": [
            "Carlos received an incomplete grade.",
            "Carlos must retake the course.",
            "Carlos has been removed from the graduation list.",
        ],
    },
    {
        "id": "hs_ext_05", "pattern": "hypothetical_syllogism",
        "seeds": [
            "If a production line halts unexpectedly, quality control is suspended.",
            "If quality control is suspended, all outgoing shipments are placed on hold.",
            "If shipments are placed on hold, clients are notified of a delay.",
            "If clients are notified of a delay, the sales team logs a service disruption.",
            "The production line has halted unexpectedly.",
        ],
        "goal": "The sales team has logged a service disruption.",
        "steps": [
            "Quality control has been suspended.",
            "All outgoing shipments have been placed on hold.",
            "Clients have been notified of a delay.",
            "The sales team has logged a service disruption.",
        ],
    },
    {
        "id": "hs_ext_06", "pattern": "hypothetical_syllogism",
        "seeds": [
            "If a patient tests positive for a notifiable disease, the public health authority must be informed.",
            "If the public health authority is informed, a contact tracing investigation is initiated.",
            "Patient Ward has tested positive for a notifiable disease.",
        ],
        "goal": "A contact tracing investigation has been initiated.",
        "steps": [
            "The public health authority has been informed.",
            "A contact tracing investigation has been initiated.",
        ],
    },

    # ── DISJUNCTIVE SYLLOGISM EXTENDED ───────────────────────────────────────
    # Each chain has consequence rules present in seeds so the model must
    # do DS elimination in context of a longer premise set — targets B3.

    {
        "id": "ds_ext_01", "pattern": "disjunctive_syllogism",
        "seeds": [
            "The trial failure was due to either patient non-compliance or dosage miscalculation.",
            "Patient compliance was verified through electronic pill monitoring throughout the trial.",
            "If the failure was due to dosage miscalculation, the pharmacy must be audited.",
            "If the pharmacy is audited, all affected batches must be recalled.",
        ],
        "goal": "All affected batches must be recalled.",
        "steps": [
            "The trial failure was not due to patient non-compliance.",
            "The trial failure was due to dosage miscalculation.",
            "The pharmacy must be audited.",
            "All affected batches must be recalled.",
        ],
    },
    {
        "id": "ds_ext_02", "pattern": "disjunctive_syllogism",
        "seeds": [
            "The fire started from either an electrical fault or arson.",
            "A full electrical inspection found all wiring and circuits to be in proper working order.",
            "An arson finding requires notification of the fire marshal.",
            "Notification of the fire marshal triggers a formal criminal investigation.",
        ],
        "goal": "A formal criminal investigation has been triggered.",
        "steps": [
            "The fire was not started by an electrical fault.",
            "The fire was started by arson.",
            "The fire marshal has been notified.",
            "A formal criminal investigation has been triggered.",
        ],
    },
    {
        "id": "ds_ext_03", "pattern": "disjunctive_syllogism",
        "seeds": [
            "The product recall was initiated due to either a manufacturing defect or a contamination event.",
            "Factory audits confirmed all manufacturing processes met quality standards.",
            "A contamination event requires all products from that batch to be quarantined.",
            "Quarantined products must undergo independent laboratory testing.",
        ],
        "goal": "The products must undergo independent laboratory testing.",
        "steps": [
            "The recall was not due to a manufacturing defect.",
            "The recall was initiated due to a contamination event.",
            "All products from that batch must be quarantined.",
            "The products must undergo independent laboratory testing.",
        ],
    },
    {
        "id": "ds_ext_04", "pattern": "disjunctive_syllogism",
        "seeds": [
            "The plagiarism was either intentional or the result of improper citation practices.",
            "The student submitted a notarized affidavit confirming no intent to deceive.",
            "Improper citation practices require mandatory academic integrity training.",
        ],
        "goal": "The student must complete mandatory academic integrity training.",
        "steps": [
            "The plagiarism was not intentional.",
            "The plagiarism was the result of improper citation practices.",
            "The student must complete mandatory academic integrity training.",
        ],
    },
    {
        "id": "ds_ext_05", "pattern": "disjunctive_syllogism",
        "seeds": [
            "The data exfiltration occurred through either a phishing attack or an insider threat.",
            "All employee accounts showed normal login patterns with no phishing indicators.",
            "An insider threat requires immediate suspension of the suspect's system access.",
        ],
        "goal": "The suspect's system access has been suspended.",
        "steps": [
            "The data exfiltration did not occur through a phishing attack.",
            "The data exfiltration occurred through an insider threat.",
            "The suspect's system access has been immediately suspended.",
        ],
    },
    {
        "id": "ds_ext_06", "pattern": "disjunctive_syllogism",
        "seeds": [
            "The roof damage was caused by either storm impact or material deterioration.",
            "Meteorological records show no severe weather events during the damage period.",
            "Material deterioration voids the manufacturer's warranty on the roofing.",
            "A voided warranty requires the property owner to bear full repair costs.",
        ],
        "goal": "The property owner must bear full repair costs.",
        "steps": [
            "The roof damage was not caused by storm impact.",
            "The roof damage was caused by material deterioration.",
            "The manufacturer's warranty on the roofing has been voided.",
            "The property owner must bear full repair costs.",
        ],
    },

    # ── CONTRAPOSITIVE CHAIN EXTENDED ────────────────────────────────────────

    {
        "id": "cp_ext_01", "pattern": "contrapositive_chain",
        "seeds": [
            "All licensed food handlers must hold a valid food hygiene certificate.",
            "Anyone with a valid food hygiene certificate has completed the food safety course.",
            "Chef Moreau has not completed the food safety course.",
        ],
        "goal": "Chef Moreau is not a licensed food handler.",
        "steps": [
            "If someone has not completed the food safety course, they do not hold a valid food hygiene certificate.",
            "Chef Moreau does not hold a valid food hygiene certificate.",
            "If someone does not hold a valid food hygiene certificate, they are not a licensed food handler.",
            "Chef Moreau is not a licensed food handler.",
        ],
    },
    {
        "id": "cp_ext_02", "pattern": "contrapositive_chain",
        "seeds": [
            "Any accredited hospital has received approval from the national health authority.",
            "Any facility approved by the national health authority has met minimum staffing requirements.",
            "Any facility that meets minimum staffing requirements has passed a patient safety audit.",
            "Brookside Clinic has not passed a patient safety audit.",
        ],
        "goal": "Brookside Clinic is not an accredited hospital.",
        "steps": [
            "If a facility has not passed a patient safety audit, it has not met minimum staffing requirements.",
            "Brookside Clinic has not met minimum staffing requirements.",
            "If a facility has not met minimum staffing requirements, it has not been approved by the national health authority.",
            "Brookside Clinic has not been approved by the national health authority.",
            "If a facility has not been approved by the national health authority, it is not an accredited hospital.",
            "Brookside Clinic is not an accredited hospital.",
        ],
    },
    {
        "id": "cp_ext_03", "pattern": "contrapositive_chain",
        "seeds": [
            "All investment advisors are registered with the financial regulatory authority.",
            "All registered advisors have submitted a compliance declaration.",
            "Mr. Nkosi has not submitted a compliance declaration.",
        ],
        "goal": "Mr. Nkosi is not an investment advisor.",
        "steps": [
            "If someone has not submitted a compliance declaration, they are not registered with the financial regulatory authority.",
            "Mr. Nkosi is not registered with the financial regulatory authority.",
            "If someone is not registered with the financial regulatory authority, they are not an investment advisor.",
            "Mr. Nkosi is not an investment advisor.",
        ],
    },
    {
        "id": "cp_ext_04", "pattern": "contrapositive_chain",
        "seeds": [
            "All civil engineers licensed to oversee bridge construction have passed the structural engineering board exam.",
            "Anyone who has passed the structural engineering board exam has completed a supervised internship.",
            "Engineer Delacroix has not completed a supervised internship.",
        ],
        "goal": "Engineer Delacroix is not licensed to oversee bridge construction.",
        "steps": [
            "If someone has not completed a supervised internship, they have not passed the structural engineering board exam.",
            "Engineer Delacroix has not passed the structural engineering board exam.",
            "If someone has not passed the structural engineering board exam, they are not licensed to oversee bridge construction.",
            "Engineer Delacroix is not licensed to oversee bridge construction.",
        ],
    },
    {
        "id": "cp_ext_05", "pattern": "contrapositive_chain",
        "seeds": [
            "All peer-reviewed journals are indexed in academic databases.",
            "All journals indexed in academic databases have an assigned ISSN.",
            "The Quarterly Review of Metaphysics does not have an assigned ISSN.",
        ],
        "goal": "The Quarterly Review of Metaphysics is not a peer-reviewed journal.",
        "steps": [
            "If a journal does not have an assigned ISSN, it is not indexed in academic databases.",
            "The Quarterly Review of Metaphysics is not indexed in academic databases.",
            "If a journal is not indexed in academic databases, it is not a peer-reviewed journal.",
            "The Quarterly Review of Metaphysics is not a peer-reviewed journal.",
        ],
    },

    # ── MIXED NEGATION EXTENDED ──────────────────────────────────────────────
    # Pattern: contrapositive derivation feeding a forward consequence chain.

    {
        "id": "mx_neg_01", "pattern": "mixed_negation",
        "seeds": [
            "All policyholders with a valid claim have submitted the required documentation.",
            "Any policyholder without a valid claim cannot receive a payout.",
            "Mr. Huang has not submitted the required documentation.",
            "Any policyholder who cannot receive a payout will have their case closed.",
            "A closed case requires the policyholder to reapply from the beginning.",
        ],
        "goal": "Mr. Huang must reapply from the beginning.",
        "steps": [
            "If someone has not submitted the required documentation, they do not have a valid claim.",
            "Mr. Huang does not have a valid claim.",
            "Mr. Huang cannot receive a payout.",
            "Mr. Huang's case will be closed.",
            "Mr. Huang must reapply from the beginning.",
        ],
    },
    {
        "id": "mx_neg_02", "pattern": "mixed_negation",
        "seeds": [
            "All accredited laboratories have undergone an ISO 17025 assessment.",
            "Laboratory results from non-accredited labs are inadmissible in regulatory proceedings.",
            "The Hillcroft Lab has not undergone an ISO 17025 assessment.",
            "Inadmissible results must be retested by a certified laboratory.",
        ],
        "goal": "The Hillcroft Lab's results must be retested by a certified laboratory.",
        "steps": [
            "If a laboratory has not undergone an ISO 17025 assessment, it is not accredited.",
            "The Hillcroft Lab is not accredited.",
            "The Hillcroft Lab's results are inadmissible in regulatory proceedings.",
            "The Hillcroft Lab's results must be retested by a certified laboratory.",
        ],
    },
    {
        "id": "mx_neg_03", "pattern": "mixed_negation",
        "seeds": [
            "All active commercial pilots have passed a biannual medical examination.",
            "Any pilot who has not passed the medical examination is grounded.",
            "Captain Faber has not passed a biannual medical examination.",
            "Any grounded pilot is removed from the active duty roster.",
            "Any pilot removed from the active duty roster must complete a fitness-for-duty review.",
        ],
        "goal": "Captain Faber must complete a fitness-for-duty review.",
        "steps": [
            "If a pilot has not passed the medical examination, they are not an active commercial pilot.",
            "Captain Faber is not an active commercial pilot.",
            "Captain Faber has been grounded.",
            "Captain Faber has been removed from the active duty roster.",
            "Captain Faber must complete a fitness-for-duty review.",
        ],
    },
    {
        "id": "mx_neg_04", "pattern": "mixed_negation",
        "seeds": [
            "All tenured professors have published at least ten peer-reviewed articles.",
            "Any faculty member without tenure cannot supervise doctoral candidates.",
            "Professor Lindqvist has not published ten peer-reviewed articles.",
            "Faculty who cannot supervise doctoral candidates are ineligible for research grants.",
        ],
        "goal": "Professor Lindqvist is ineligible for research grants.",
        "steps": [
            "If a professor has not published at least ten peer-reviewed articles, they are not tenured.",
            "Professor Lindqvist is not a tenured professor.",
            "Professor Lindqvist cannot supervise doctoral candidates.",
            "Professor Lindqvist is ineligible for research grants.",
        ],
    },

    # ── LONG FORWARD CHAIN EXTENDED ──────────────────────────────────────────
    # 5-step chains in domains distinct from the climate treaty (B6).
    # Entity fact is the LAST seed so the model sees it alongside the rules.

    {
        "id": "lfc_ext_01", "pattern": "long_forward_chain",
        "seeds": [
            "Any pharmaceutical compound that completes preclinical trials is eligible for Phase 1 human trials.",
            "Any compound that completes Phase 1 trials advances to Phase 2 efficacy testing.",
            "Any compound that demonstrates efficacy in Phase 2 advances to Phase 3 large-scale trials.",
            "Any compound that completes Phase 3 trials is reviewed by the regulatory authority.",
            "Any compound reviewed by the regulatory authority and found safe is granted market approval.",
            "Compound Zylan has completed preclinical trials.",
        ],
        "goal": "Compound Zylan has been granted market approval.",
        "steps": [
            "Compound Zylan is eligible for Phase 1 human trials.",
            "Compound Zylan has advanced to Phase 2 efficacy testing.",
            "Compound Zylan has advanced to Phase 3 large-scale trials.",
            "Compound Zylan is under review by the regulatory authority.",
            "Compound Zylan has been granted market approval.",
        ],
    },
    {
        "id": "lfc_ext_02", "pattern": "long_forward_chain",
        "seeds": [
            "Any city that achieves smart city certification attracts technology investment.",
            "Any city that attracts technology investment experiences growth in the tech workforce.",
            "Any city with a growing tech workforce sees rising demand for housing.",
            "Any city with rising housing demand must expand its public transport network.",
            "Any city that expands its public transport network reduces commute times.",
            "Northfield has achieved smart city certification.",
        ],
        "goal": "Northfield will see reduced commute times.",
        "steps": [
            "Northfield attracts technology investment.",
            "Northfield experiences growth in the tech workforce.",
            "Northfield sees rising demand for housing.",
            "Northfield must expand its public transport network.",
            "Northfield will see reduced commute times.",
        ],
    },
    {
        "id": "lfc_ext_03", "pattern": "long_forward_chain",
        "seeds": [
            "Any supplier that achieves ISO certification is approved for the preferred vendor list.",
            "Any vendor on the preferred vendor list is invited to participate in tender bids.",
            "Any vendor invited to bid and meeting specifications is awarded a contract.",
            "Any vendor awarded a contract must complete an onboarding assessment.",
            "Any vendor that completes onboarding is assigned a dedicated procurement manager.",
            "Hartwell Components has achieved ISO certification.",
        ],
        "goal": "Hartwell Components has been assigned a dedicated procurement manager.",
        "steps": [
            "Hartwell Components is approved for the preferred vendor list.",
            "Hartwell Components is invited to participate in tender bids.",
            "Hartwell Components is awarded a contract.",
            "Hartwell Components must complete an onboarding assessment.",
            "Hartwell Components has been assigned a dedicated procurement manager.",
        ],
    },
    {
        "id": "lfc_ext_04", "pattern": "long_forward_chain",
        "seeds": [
            "Any athlete who meets the qualifying standard is selected for the national trials.",
            "Any athlete selected for the national trials competes in the selection event.",
            "Any athlete who wins the selection event earns a place on the national team.",
            "Any athlete on the national team receives government athletic funding.",
            "Any athlete with government funding can access elite training facilities.",
            "Rashida has met the qualifying standard.",
        ],
        "goal": "Rashida can access elite training facilities.",
        "steps": [
            "Rashida has been selected for the national trials.",
            "Rashida competes in the selection event.",
            "Rashida earns a place on the national team.",
            "Rashida receives government athletic funding.",
            "Rashida can access elite training facilities.",
        ],
    },
]
