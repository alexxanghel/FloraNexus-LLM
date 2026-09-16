from __future__ import annotations

from typing import Any

from neo4j import GraphDatabase

from floranexus_llm.settings import settings


def normalize_text(value: str) -> str:
    text = (value or "").strip().casefold()
    text = text.replace("_", " ").replace("-", " ")
    return " ".join(text.split())


class Neo4jFloraStore:
    def __init__(self) -> None:
        self.driver = GraphDatabase.driver(
            settings.neo4j_uri,
            auth=(settings.neo4j_username, settings.neo4j_password.get_secret_value()),
        )
        self.database = settings.neo4j_database

    def verify_connectivity(self) -> None:
        self.driver.verify_connectivity()

    def close(self) -> None:
        self.driver.close()

    def _run(self, cypher: str, **params: Any) -> list[dict[str, Any]]:
        with self.driver.session(database=self.database) as session:
            result = session.run(cypher, **params)
            return [dict(record) for record in result]

    @staticmethod
    def _clean_text(value: Any) -> str:
        return str(value).strip() if value is not None else ""

    @staticmethod
    def _clean_list(values: list[Any]) -> list[str]:
        out: list[str] = []
        seen = set()
        for value in values:
            text = str(value).strip() if value is not None else ""
            if not text:
                continue
            key = text.casefold()
            if key in seen:
                continue
            seen.add(key)
            out.append(text)
        return out

    @staticmethod
    def _filter_fact_items(items: list[dict[str, Any]], text_key: str = "text") -> list[dict[str, Any]]:
        cleaned: list[dict[str, Any]] = []
        seen = set()

        for item in items:
            text = str(item.get(text_key, "") or "").strip()
            if not text:
                continue

            key = text.casefold()
            if key in seen:
                continue
            seen.add(key)

            cleaned.append(
                {
                    k: v
                    for k, v in item.items()
                    if v not in ("", None, [], {})
                }
            )

        return cleaned

    def _load_all_plants(self) -> list[dict[str, Any]]:
        cypher = """
        MATCH (p:Plant)
        OPTIONAL MATCH (p)-[:HAS_ALIAS]->(a:Alias)
        RETURN
          p.plant_id AS plant_id,
          coalesce(p.common_name_ro, "") AS common_name_ro,
          coalesce(p.common_name_en, "") AS common_name_en,
          coalesce(p.scientific_name, "") AS scientific_name,
          coalesce(p.family, "") AS family,
          coalesce(p.notes_ro, "") AS notes_ro,
          [x IN collect(DISTINCT a.alias_value) WHERE x IS NOT NULL AND trim(x) <> ""] AS aliases
        """
        rows = self._run(cypher)

        plants: list[dict[str, Any]] = []
        for row in rows:
            plants.append(
                {
                    "plant_id": self._clean_text(row.get("plant_id")),
                    "common_name_ro": self._clean_text(row.get("common_name_ro")),
                    "common_name_en": self._clean_text(row.get("common_name_en")),
                    "scientific_name": self._clean_text(row.get("scientific_name")),
                    "family": self._clean_text(row.get("family")),
                    "notes_ro": self._clean_text(row.get("notes_ro")),
                    "aliases": self._clean_list(row.get("aliases") or []),
                }
            )
        return plants

    def resolve_detected_plant(
        self,
        *,
        predicted_plant_id: str,
        predicted_name_ro: str,
        predicted_scientific_name: str,
        top_k_ids: list[str],
    ) -> dict[str, Any]:
        plants = self._load_all_plants()

        search_terms: list[tuple[str, str]] = []

        def add_term(value: str, source: str) -> None:
            norm = normalize_text(value)
            if norm:
                search_terms.append((norm, source))

        add_term(predicted_scientific_name, "predicted_scientific_name")
        add_term(predicted_name_ro, "predicted_name_ro")
        add_term(predicted_plant_id, "predicted_plant_id")
        add_term(predicted_plant_id.replace("_", " "), "predicted_plant_id_slug")

        for item in top_k_ids:
            add_term(item, "top_k_plant_id")
            add_term(item.replace("_", " "), "top_k_plant_id_slug")

        best_match: dict[str, Any] | None = None
        best_score = -1
        best_source = ""

        for plant in plants:
            scientific = normalize_text(plant["scientific_name"])
            common_ro = normalize_text(plant["common_name_ro"])
            common_en = normalize_text(plant["common_name_en"])
            plant_id_norm = normalize_text(plant["plant_id"])
            aliases = [normalize_text(alias) for alias in plant["aliases"]]

            for term, source in search_terms:
                score = -1

                if term and term == scientific:
                    score = 100
                elif term and term == common_ro:
                    score = 95
                elif term and term == common_en:
                    score = 92
                elif term and term == plant_id_norm:
                    score = 90
                elif term and term in aliases:
                    score = 85
                elif term and scientific and term in scientific:
                    score = 60
                elif term and common_ro and term in common_ro:
                    score = 55

                if score > best_score:
                    best_score = score
                    best_match = plant
                    best_source = source

        if best_match is None:
            return {
                "resolved": False,
                "matched_by": "",
                "plant": {
                    "graph_plant_id": "",
                    "common_name_ro": predicted_name_ro,
                    "scientific_name": predicted_scientific_name,
                    "family": "",
                    "notes_ro": "",
                    "aliases": [],
                },
            }

        return {
            "resolved": True,
            "matched_by": best_source,
            "plant": {
                "graph_plant_id": best_match["plant_id"],
                "common_name_ro": best_match["common_name_ro"] or predicted_name_ro,
                "scientific_name": best_match["scientific_name"] or predicted_scientific_name,
                "family": best_match["family"],
                "notes_ro": best_match["notes_ro"],
                "aliases": best_match["aliases"],
            },
        }

    def get_plant_identity(self, *, graph_plant_id: str) -> dict[str, Any]:
        cypher = """
        MATCH (p:Plant {plant_id: $plant_id})
        OPTIONAL MATCH (p)-[:HAS_ALIAS]->(a:Alias)
        RETURN
          p.plant_id AS plant_id,
          coalesce(p.common_name_ro, "") AS common_name_ro,
          coalesce(p.common_name_en, "") AS common_name_en,
          coalesce(p.scientific_name, "") AS scientific_name,
          coalesce(p.family, "") AS family,
          coalesce(p.notes_ro, "") AS notes_ro,
          [x IN collect(DISTINCT a.alias_value) WHERE x IS NOT NULL AND trim(x) <> ""] AS aliases
        """
        rows = self._run(cypher, plant_id=graph_plant_id)

        if not rows:
            return {
                "found": False,
                "plant": {
                    "graph_plant_id": graph_plant_id,
                    "common_name_ro": "",
                    "common_name_en": "",
                    "scientific_name": "",
                    "family": "",
                    "notes_ro": "",
                    "aliases": [],
                },
            }

        row = rows[0]
        return {
            "found": True,
            "plant": {
                "graph_plant_id": self._clean_text(row.get("plant_id")),
                "common_name_ro": self._clean_text(row.get("common_name_ro")),
                "common_name_en": self._clean_text(row.get("common_name_en")),
                "scientific_name": self._clean_text(row.get("scientific_name")),
                "family": self._clean_text(row.get("family")),
                "notes_ro": self._clean_text(row.get("notes_ro")),
                "aliases": self._clean_list(row.get("aliases") or []),
            },
        }

    def get_plant_benefits(self, *, graph_plant_id: str) -> dict[str, Any]:
        identity = self.get_plant_identity(graph_plant_id=graph_plant_id)

        cypher = """
        MATCH (p:Plant {plant_id: $plant_id})-[:HAS_BENEFIT]->(b:Benefit)
        WHERE coalesce(b.chatbot_visible, true) = true
          AND coalesce(b.record_status, "active") <> "hidden"
        RETURN collect(DISTINCT {
          text: coalesce(b.benefit_text_ro, b.benefit_text_en, ""),
          benefit_type: coalesce(b.benefit_type, ""),
          evidence_level: coalesce(b.evidence_level, ""),
          authority: coalesce(b.authority, ""),
          citation_note: coalesce(b.citation_note, "")
        }) AS items
        """
        rows = self._run(cypher, plant_id=graph_plant_id)
        items = rows[0]["items"] if rows else []

        return {
            "plant": identity["plant"],
            "benefits": self._filter_fact_items(items),
        }

    def get_plant_uses(self, *, graph_plant_id: str) -> dict[str, Any]:
        identity = self.get_plant_identity(graph_plant_id=graph_plant_id)

        cypher = """
        MATCH (p:Plant {plant_id: $plant_id})-[:HAS_USE]->(u:Use)
        WHERE coalesce(u.chatbot_visible, true) = true
          AND coalesce(u.record_status, "active") <> "hidden"
        RETURN collect(DISTINCT {
          text: coalesce(u.use_text_ro, u.use_text_en, ""),
          use_type: coalesce(u.use_type, ""),
          evidence_level: coalesce(u.evidence_level, ""),
          authority: coalesce(u.authority, ""),
          citation_note: coalesce(u.citation_note, "")
        }) AS items
        """
        rows = self._run(cypher, plant_id=graph_plant_id)
        items = rows[0]["items"] if rows else []

        return {
            "plant": identity["plant"],
            "uses": self._filter_fact_items(items),
        }

    def get_plant_contraindications(self, *, graph_plant_id: str) -> dict[str, Any]:
        identity = self.get_plant_identity(graph_plant_id=graph_plant_id)

        cypher_contra = """
        MATCH (p:Plant {plant_id: $plant_id})-[:HAS_CONTRAINDICATION]->(c:Contraindication)
        WHERE coalesce(c.chatbot_visible, true) = true
          AND coalesce(c.record_status, "active") <> "hidden"
        RETURN collect(DISTINCT {
          category: "contraindication",
          text: coalesce(c.contra_text_ro, c.contra_text_en, ""),
          evidence_level: coalesce(c.evidence_level, ""),
          authority: coalesce(c.authority, ""),
          citation_note: coalesce(c.citation_note, "")
        }) AS items
        """

        cypher_warning = """
        MATCH (p:Plant {plant_id: $plant_id})-[:HAS_WARNING]->(w:Warning)
        WHERE coalesce(w.chatbot_visible, true) = true
          AND coalesce(w.record_status, "active") <> "hidden"
        RETURN collect(DISTINCT {
          category: "warning",
          text: coalesce(w.warning_text_ro, w.warning_text_en, ""),
          severity: coalesce(w.severity, ""),
          evidence_level: coalesce(w.evidence_level, ""),
          authority: coalesce(w.authority, ""),
          citation_note: coalesce(w.citation_note, "")
        }) AS items
        """

        cypher_effect = """
        MATCH (p:Plant {plant_id: $plant_id})-[:HAS_ADVERSE_EFFECT]->(e:AdverseEffect)
        WHERE coalesce(e.chatbot_visible, true) = true
          AND coalesce(e.record_status, "active") <> "hidden"
        RETURN collect(DISTINCT {
          category: "adverse_effect",
          text: coalesce(e.effect_text_ro, e.effect_text_en, ""),
          severity: coalesce(e.severity, ""),
          evidence_level: coalesce(e.evidence_level, ""),
          authority: coalesce(e.authority, ""),
          citation_note: coalesce(e.citation_note, "")
        }) AS items
        """

        contra = self._run(cypher_contra, plant_id=graph_plant_id)
        warnings = self._run(cypher_warning, plant_id=graph_plant_id)
        effects = self._run(cypher_effect, plant_id=graph_plant_id)

        items = []
        items.extend(contra[0]["items"] if contra else [])
        items.extend(warnings[0]["items"] if warnings else [])
        items.extend(effects[0]["items"] if effects else [])

        return {
            "plant": identity["plant"],
            "contraindications": self._filter_fact_items(items),
        }

    def get_plant_interactions(self, *, graph_plant_id: str) -> dict[str, Any]:
        identity = self.get_plant_identity(graph_plant_id=graph_plant_id)

        cypher = """
        MATCH (p:Plant {plant_id: $plant_id})-[:INTERACTS_WITH]->(i:Interaction)
        WHERE coalesce(i.chatbot_visible, true) = true
          AND coalesce(i.record_status, "active") <> "hidden"
        RETURN collect(DISTINCT {
          text: coalesce(i.interaction_text_ro, i.interaction_text_en, ""),
          severity: coalesce(i.severity, ""),
          evidence_level: coalesce(i.evidence_level, ""),
          authority: coalesce(i.authority, ""),
          citation_note: coalesce(i.citation_note, "")
        }) AS items
        """
        rows = self._run(cypher, plant_id=graph_plant_id)
        items = rows[0]["items"] if rows else []

        return {
            "plant": identity["plant"],
            "interactions": self._filter_fact_items(items),
        }

    def get_plant_usable_parts(self, *, graph_plant_id: str) -> dict[str, Any]:
        identity = self.get_plant_identity(graph_plant_id=graph_plant_id)

        cypher = """
        MATCH (p:Plant {plant_id: $plant_id})-[:USES_PART]->(part:PartUsed)
        WHERE coalesce(part.chatbot_visible, true) = true
          AND coalesce(part.record_status, "active") <> "hidden"
        RETURN collect(DISTINCT {
          part_name_ro: coalesce(part.part_name_ro, part.part_name_en, ""),
          notes_ro: coalesce(part.notes_ro, ""),
          evidence_level: coalesce(part.evidence_level, ""),
          authority: coalesce(part.authority, ""),
          citation_note: coalesce(part.citation_note, "")
        }) AS items
        """
        rows = self._run(cypher, plant_id=graph_plant_id)
        items = rows[0]["items"] if rows else []

        cleaned: list[dict[str, Any]] = []
        seen = set()
        for item in items:
            part_name = self._clean_text(item.get("part_name_ro"))
            if not part_name:
                continue
            key = part_name.casefold()
            if key in seen:
                continue
            seen.add(key)
            cleaned.append({k: v for k, v in item.items() if v not in ("", None, [], {})})

        return {
            "plant": identity["plant"],
            "usable_parts": cleaned,
        }

    def resolve_plant_candidates(self, *, plant_ids: list[str]) -> dict[str, Any]:
        plants = self._load_all_plants()
        by_norm: dict[str, dict[str, Any]] = {}

        for plant in plants:
            values = [
                plant["plant_id"],
                plant["common_name_ro"],
                plant["common_name_en"],
                plant["scientific_name"],
                *plant["aliases"],
            ]
            for value in values:
                norm = normalize_text(value)
                if norm and norm not in by_norm:
                    by_norm[norm] = plant

        candidates: list[dict[str, Any]] = []
        for pid in plant_ids:
            norm_candidates = [
                normalize_text(pid),
                normalize_text(pid.replace("_", " ")),
            ]

            match = None
            for norm in norm_candidates:
                if norm in by_norm:
                    match = by_norm[norm]
                    break

            if match is None:
                pretty = pid.replace("_", " ").strip().title()
                candidates.append(
                    {
                        "graph_plant_id": "",
                        "plant_id": pid,
                        "plant_name_ro": pretty,
                        "plant_name_scientific": pretty,
                        "aliases": [],
                    }
                )
            else:
                candidates.append(
                    {
                        "graph_plant_id": match["plant_id"],
                        "plant_id": pid,
                        "plant_name_ro": match["common_name_ro"] or match["plant_id"],
                        "plant_name_scientific": match["scientific_name"] or match["plant_id"],
                        "aliases": match["aliases"],
                    }
                )

        return {"candidates": candidates}