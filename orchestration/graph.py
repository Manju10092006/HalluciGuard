async def _verifier_node(state: HalluciGuardState) -> dict[str, Any]:
    VerificationPipeline, SuspiciousClaim, VerifierInputV2 = _verifier_imports()
    node_start = start_timer()

    try:
        # ---------------------------------------------------------
        # 1. Extract claims from the DRAFT ANSWER
        # ---------------------------------------------------------
        # The user query is context only.
        # The verifier must verify what the model actually claimed.
        draft_response = (
            state.get("llm_response")
            or state.get("draft_response")
            or ""
        )

        if not draft_response.strip():
            raise ValueError("No draft response available for verification.")

        draft_claims = extract_claims(draft_response)

        # If claim extraction produces nothing, fail closed.
        if not draft_claims:
            raise ValueError(
                "No factual claims could be extracted from the draft response."
            )

        suspicious_claims = [
            SuspiciousClaim(
                claim_id=f"dc-{idx + 1}",
                text=claim,
            )
            for idx, claim in enumerate(draft_claims)
        ]

        # ---------------------------------------------------------
        # 2. Build verifier input
        # ---------------------------------------------------------
        payload = VerifierInputV2(
            query_id=(
                state.get("request_id")
                or state.get("execution_id")
                or str(uuid.uuid4())
            ),
            domain=state.get("domain", "general"),
            suspicious_claims=suspicious_claims,
        )

        # ---------------------------------------------------------
        # 3. Run verifier with timeout
        # ---------------------------------------------------------
        try:
            verifier_timeout = float(
                os.environ.get(
                    "VERIFIER_TIMEOUT_SECONDS",
                    "120.0",
                )
            )

            verifier_res = await asyncio.wait_for(
                VerificationPipeline().verify(payload),
                timeout=verifier_timeout,
            )

            verifier = _dump(verifier_res)

        except asyncio.TimeoutError as sub_err:
            raise RuntimeError(
                f"Verifier failed: Timeout after {verifier_timeout}s"
            ) from sub_err

        except Exception as sub_err:
            raise RuntimeError(
                f"Verifier failed: "
                f"{type(sub_err).__name__}: {sub_err}"
            ) from sub_err

        # ---------------------------------------------------------
        # 4. Process verification results
        # ---------------------------------------------------------
        judge_pairs: list[dict[str, Any]] = []
        evidence_all: list[dict[str, Any]] = []
        nli_results: list[dict[str, Any]] = []
        claims: list[dict[str, Any]] = []

        has_contradiction = False
        has_verified = False
        has_conflicted = False
        has_unverified = False

        claim_reports = verifier.get("claim_evidence") or verifier.get(
            "claim_reports",
            [],
        )

        for report in claim_reports:
            verdict_raw = str(
                report.get("verdict", "")
            ).lower().strip()

            clean_verdict = "unverified"

            # ---------------------------------------------
            # Verdict normalization
            # ---------------------------------------------
            if (
                "contradict" in verdict_raw
                or "hallucinat" in verdict_raw
            ):
                has_contradiction = True
                clean_verdict = "contradicted"

            elif (
                verdict_raw
                in {
                    "verified",
                    "supported",
                    "verdictlabel.verified",
                }
                or (
                    verdict_raw.startswith("verif")
                    and "unverif" not in verdict_raw
                )
            ):
                has_verified = True
                clean_verdict = "verified"

            elif "conflict" in verdict_raw:
                has_conflicted = True
                clean_verdict = "conflicted"

            else:
                has_unverified = True

            # ---------------------------------------------
            # Store normalized claim
            # ---------------------------------------------
            claims.append(
                {
                    "claim_id": report.get("claim_id"),
                    "text": report.get("claim_text"),
                    "verdict": clean_verdict,
                }
            )

            # ---------------------------------------------
            # Process evidence
            # ---------------------------------------------
            evidence_items = report.get("evidence", [])

            for evidence in evidence_items:
                evidence_all.append(evidence)

                entailment_score = float(
                    evidence.get(
                        "entailment_score",
                        evidence.get(
                            "nli_entailment",
                            0.0,
                        ),
                    )
                    or 0.0
                )

                contradiction_score = float(
                    evidence.get(
                        "contradiction_score",
                        evidence.get(
                            "nli_contradiction",
                            0.0,
                        ),
                    )
                    or 0.0
                )

                credibility_score = float(
                    evidence.get(
                        "credibility_score",
                        0.0,
                    )
                    or 0.0
                )

                entailment_label = str(
                    evidence.get(
                        "entailment_label",
                        "neutral",
                    )
                )

                # -----------------------------------------
                # NLI result for observability
                # -----------------------------------------
                nli_results.append(
                    {
                        "claim": report.get(
                            "claim_text",
                            "",
                        ),
                        "label": entailment_label,
                        "score": entailment_score,
                    }
                )

                # -----------------------------------------
                # Judge evidence pair
                # -----------------------------------------
                judge_pairs.append(
                    {
                        "claim": report.get(
                            "claim_text",
                            "",
                        ),
                        "evidence": evidence.get(
                            "snippet",
                            "",
                        ),
                        "source": evidence.get(
                            "source",
                            "",
                        ),
                        "url": evidence.get(
                            "url",
                            "",
                        ),
                        "entailment_label": entailment_label,
                        "entailment_score": entailment_score,
                        "contradiction_score": contradiction_score,
                        "credibility_score": credibility_score,
                        "evidence_class": classify_evidence_class(
                            entailment_label,
                            entailment_score,
                            contradiction_score,
                            credibility_score,
                        ).value,
                    }
                )

        # ---------------------------------------------------------
        # 5. Build bus message
        # ---------------------------------------------------------
        bus = add_bus_message(
            state,
            source_agent="verifier",
            target_agent="supervisor",
            message_type="VERIFICATION_RESULT",
            payload={
                "claims_count": len(claims),
                "claims_extracted": len(draft_claims),
                "evidence_count": len(evidence_all),
                "has_contradiction": has_contradiction,
                "has_verified": has_verified,
            },
        )

        # ---------------------------------------------------------
        # 6. Convert raw verifier output into canonical result
        # ---------------------------------------------------------
        canonical_verifier_result = _build_canonical_verifier_result(
            verifier,
            payload.query_id,
            payload.domain,
        )

        # ---------------------------------------------------------
        # 7. Calculate overall verification status
        # ---------------------------------------------------------
        overall_status = (
            PipelineState.CONTRADICTED.value
            if has_contradiction
            else PipelineState.CONFLICTED.value
            if has_conflicted
            else PipelineState.VERIFIED.value
            if has_verified
            else PipelineState.UNVERIFIED.value
        )

        # ---------------------------------------------------------
        # 8. Return state update
        # ---------------------------------------------------------
        return {
            "verifier": verifier,

            "verifier_result": _dump(
                canonical_verifier_result
            ),

            "claims": claims,

            # Important:
            # These are the claims extracted from the draft answer,
            # not the user's question.
            "draft_claims": draft_claims,

            "verification_summary": {
                "claims_extracted": len(draft_claims),
                "claims_verified": int(has_verified),
                "claims_contradicted": int(has_contradiction),
                "claims_conflicted": int(has_conflicted),
                "claims_unverified": int(has_unverified),
                "overall_status": overall_status,
            },

            "judge_pairs": judge_pairs,

            "evidence": evidence_all,

            "retrieved_evidence": evidence_all,

            "ranked_evidence": _dump(
                canonical_verifier_result.evidence
            ),

            "nli_results": nli_results,

            "verification_status": overall_status,

            "inter_agent_bus": bus,

            "updated_at": utc_now(),

            "trace": add_trace(
                state,
                "verifier",
                "completed",
                latency_ms=elapsed_ms(node_start),
                claim_count=len(claims),
                evidence_count=len(evidence_all),
                has_contradiction=has_contradiction,
            ),
        }

    # -------------------------------------------------------------
    # 9. Fail closed on verifier errors
    # -------------------------------------------------------------
    except Exception as exc:
        from orchestration.schemas import (
            VerifierResult as CanonicalVerifierResult,
            ExecutionStatus,
        )

        failed_res = CanonicalVerifierResult(
            query_id=(
                payload.query_id
                if "payload" in locals()
                else "q-failed"
            ),
            domain=state.get(
                "domain",
                "general",
            ),
            claim_reports=[],
            evidence=[],
            overall_confidence=0.0,
            status=ExecutionStatus.FAILED,
        )

        update = _failure_update(
            state,
            "verifier",
            exc,
            retryable=True,
        )

        update["verifier_result"] = _dump(
            failed_res
        )

        return update