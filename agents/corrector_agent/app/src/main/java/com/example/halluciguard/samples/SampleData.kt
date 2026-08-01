package com.example.halluciguard.samples

import com.example.halluciguard.model.AtomicClaim
import com.example.halluciguard.model.ClaimStatus
import com.example.halluciguard.model.EvidencePassage
import com.example.halluciguard.model.JudgeVerificationPayload
import com.example.halluciguard.model.SourceMetadata

object SampleData {

    val scenarios = listOf(
        ScenarioItem(
            id = "medical_1",
            title = "Medical & Pharmacology",
            subtitle = "Amoxicillin Dosage & Contraindications",
            payload = JudgeVerificationPayload(
                query = "What is the recommended adult dosage for Amoxicillin in acute sinusitis, and what are its key contraindications?",
                originalResponse = "Amoxicillin is recommended at 2,000 mg twice daily for 14 days. It is completely safe for patients with severe penicillin allergies and requires no renal dose adjustments. It inhibits bacterial cell wall synthesis by binding to penicillin-binding proteins.",
                claims = listOf(
                    AtomicClaim(
                        id = "c_med_1",
                        text = "Amoxicillin is recommended at 2,000 mg twice daily for 14 days.",
                        status = ClaimStatus.HALLUCINATED,
                        confidenceScore = 0.32,
                        evidenceIds = listOf("ev_med_1")
                    ),
                    AtomicClaim(
                        id = "c_med_2",
                        text = "It is completely safe for patients with severe penicillin allergies.",
                        status = ClaimStatus.CONTRADICTED,
                        confidenceScore = 0.15,
                        evidenceIds = listOf("ev_med_2")
                    ),
                    AtomicClaim(
                        id = "c_med_3",
                        text = "Requires no renal dose adjustments.",
                        status = ClaimStatus.HALLUCINATED,
                        confidenceScore = 0.28,
                        evidenceIds = listOf("ev_med_3")
                    ),
                    AtomicClaim(
                        id = "c_med_4",
                        text = "It inhibits bacterial cell wall synthesis by binding to penicillin-binding proteins.",
                        status = ClaimStatus.VERIFIED,
                        confidenceScore = 0.99,
                        evidenceIds = listOf("ev_med_4")
                    )
                ),
                supportingEvidence = listOf(
                    EvidencePassage(
                        id = "ev_med_1",
                        sourceTitle = "FDA Drug Label - Amoxicillin",
                        passageText = "The standard adult dosage for mild-to-moderate bacterial infections is 500 mg every 8 hours or 875 mg every 12 hours for 7 to 10 days."
                    ),
                    EvidencePassage(
                        id = "ev_med_2",
                        sourceTitle = "Clinical Pharmacology Guide 2025",
                        passageText = "Amoxicillin is strictly contraindicated in patients with a history of severe immediate hypersensitivity reactions (e.g., anaphylaxis) to penicillins or beta-lactams."
                    ),
                    EvidencePassage(
                        id = "ev_med_3",
                        sourceTitle = "Renal Dosing Guidelines",
                        passageText = "Dose reduction or extended dosing interval is required in patients with severe renal impairment (GFR < 30 mL/min)."
                    ),
                    EvidencePassage(
                        id = "ev_med_4",
                        sourceTitle = "Microbiology Text",
                        passageText = "Amoxicillin exerts bactericidal activity by inhibiting bacterial cell wall peptidoglycan synthesis via binding to penicillin-binding proteins."
                    )
                ),
                contradictionEvidence = listOf(
                    EvidencePassage(
                        id = "ev_med_2_contra",
                        sourceTitle = "Allergy & Clinical Immunology Standards",
                        passageText = "Beta-lactam antibiotics cause life-threatening allergic reactions in sensitized individuals and must never be administered."
                    )
                ),
                trustScore = 0.42,
                sourceMetadata = listOf(
                    SourceMetadata(id = "s1", title = "FDA Drug Safety Database", authorOrOrg = "US FDA"),
                    SourceMetadata(id = "s2", title = "Clinical Pharmacology Manual", authorOrOrg = "Medical Publishing")
                ),
                correctionInstructions = "Rewrite claim c_med_1 using standard FDA 875mg dosing evidence. Correct claim c_med_2 to highlight severe allergy contraindication. Update c_med_3 to note GFR-based renal dosage adjustments. Preserve c_med_4 mechanism of action exactly."
            )
        ),
        ScenarioItem(
            id = "financial_1",
            title = "Financial & Compliance Audit",
            subtitle = "Acme Corp EBITDA & Debt Covenant Cap",
            payload = JudgeVerificationPayload(
                query = "Summarize Acme Corp's Q3 EBITDA performance and debt covenant compliance status.",
                originalResponse = "Acme Corp reported a 45% increase in EBITDA reaching $120 million in Q3. The debt-to-EBITDA covenant cap is set at 5.5x, and Acme Corp was granted a permanent SEC exemption from quarterly filing. Operating expenses decreased by 3% due to supply chain optimizations.",
                claims = listOf(
                    AtomicClaim(
                        id = "c_fin_1",
                        text = "Acme Corp reported a 45% increase in EBITDA reaching $120 million in Q3.",
                        status = ClaimStatus.HALLUCINATED,
                        confidenceScore = 0.38,
                        evidenceIds = listOf("ev_fin_1")
                    ),
                    AtomicClaim(
                        id = "c_fin_2",
                        text = "The debt-to-EBITDA covenant cap is set at 5.5x.",
                        status = ClaimStatus.CONTRADICTED,
                        confidenceScore = 0.22,
                        evidenceIds = listOf("ev_fin_2")
                    ),
                    AtomicClaim(
                        id = "c_fin_3",
                        text = "Acme Corp was granted a permanent SEC exemption from quarterly filing.",
                        status = ClaimStatus.INSUFFICIENT_EVIDENCE,
                        confidenceScore = 0.10,
                        evidenceIds = emptyList()
                    ),
                    AtomicClaim(
                        id = "c_fin_4",
                        text = "Operating expenses decreased by 3% due to supply chain optimizations.",
                        status = ClaimStatus.VERIFIED,
                        confidenceScore = 0.96,
                        evidenceIds = listOf("ev_fin_4")
                    )
                ),
                supportingEvidence = listOf(
                    EvidencePassage(
                        id = "ev_fin_1",
                        sourceTitle = "Acme Corp 10-Q Filing Q3 2025",
                        passageText = "Acme Corp generated $84 million in adjusted EBITDA for Q3 2025, representing a 12% year-over-year increase."
                    ),
                    EvidencePassage(
                        id = "ev_fin_2",
                        sourceTitle = "Credit Agreement Amendment 2025",
                        passageText = "The maximum permitted total leverage ratio (Debt-to-EBITDA covenant cap) remains fixed at 3.5x."
                    ),
                    EvidencePassage(
                        id = "ev_fin_4",
                        sourceTitle = "Acme Corp Investor Presentation",
                        passageText = "Operating expenses declined by 3% YoY driven by automated logistics and warehouse supply chain efficiency."
                    )
                ),
                trustScore = 0.48,
                sourceMetadata = listOf(
                    SourceMetadata(id = "s_fin_1", title = "SEC EDGAR 10-Q Filing", authorOrOrg = "SEC")
                ),
                correctionInstructions = "Correct Q3 EBITDA figure to $84 million (12% YoY growth). Correct covenant cap to 3.5x. Replace unsupported SEC exemption statement with explicit disclaimer. Preserve operating expense reduction verbatim."
            )
        ),
        ScenarioItem(
            id = "tech_1",
            title = "Software Architecture",
            subtitle = "Rust Memory Management & GC Claims",
            payload = JudgeVerificationPayload(
                query = "How does Rust handle memory management and garbage collection?",
                originalResponse = "Rust relies on a mark-and-sweep garbage collector running every 10ms to clean unused heap memory. Variables are allocated on the JVM heap by default. RAII (Resource Acquisition Is Initialization) guarantees that memory is released when variables leave scope.",
                claims = listOf(
                    AtomicClaim(
                        id = "c_tech_1",
                        text = "Rust relies on a mark-and-sweep garbage collector running every 10ms to clean unused heap memory.",
                        status = ClaimStatus.CONTRADICTED,
                        confidenceScore = 0.11,
                        evidenceIds = listOf("ev_tech_1")
                    ),
                    AtomicClaim(
                        id = "c_tech_2",
                        text = "Variables are allocated on the JVM heap by default.",
                        status = ClaimStatus.HALLUCINATED,
                        confidenceScore = 0.20,
                        evidenceIds = listOf("ev_tech_2")
                    ),
                    AtomicClaim(
                        id = "c_tech_3",
                        text = "RAII (Resource Acquisition Is Initialization) guarantees that memory is released when variables leave scope.",
                        status = ClaimStatus.VERIFIED,
                        confidenceScore = 0.99,
                        evidenceIds = listOf("ev_tech_3")
                    )
                ),
                supportingEvidence = listOf(
                    EvidencePassage(
                        id = "ev_tech_1",
                        sourceTitle = "The Rust Programming Language Book",
                        passageText = "Rust does not use a runtime garbage collector; instead, memory is managed through an ownership model with strict compiler borrow checker rules."
                    ),
                    EvidencePassage(
                        id = "ev_tech_2",
                        sourceTitle = "Rust Reference Manual",
                        passageText = "Values in Rust are allocated on the stack by default, unless explicitly wrapped in heap-allocated smart pointers such as Box<T> or Vec<T>."
                    ),
                    EvidencePassage(
                        id = "ev_tech_3",
                        sourceTitle = "Rust Documentation",
                        passageText = "Rust enforces RAII pattern so that when an owner variable goes out of scope, its drop function is called and resources are freed instantly."
                    )
                ),
                trustScore = 0.35,
                sourceMetadata = listOf(
                    SourceMetadata(id = "s_tech_1", title = "Rust Foundation Docs", authorOrOrg = "Rust Team")
                ),
                correctionInstructions = "Eliminate all mention of GC and JVM heap. State clearly that Rust uses compile-time ownership without garbage collection, stack allocation by default, and RAII automatic drop mechanism."
            )
        )
    )
}

data class ScenarioItem(
    val id: String,
    val title: String,
    val subtitle: String,
    val payload: JudgeVerificationPayload
)
