# Caretaker cast-quality proposal contract

Caretaker writes one complete `cast_context_resolution_proposals.json` snapshot by atomic rename. The producer reads it only at a cast batch boundary. It never mutates the input, registry, aliases, source, or media from this file.

```json
{
  "version": 1,
  "source_sha256": "<full source sha256>",
  "proposals": [{
    "proposal_id": "immutable unique id",
    "registry_id": "active non-narrator actor id",
    "kind": "country | garble | duplicate_actor",
    "status": "pending | resolved",
    "scope": {
      "chapter": "NN-part_NN.txt",
      "chapter_sha256": "...",
      "unit_id": "c00s00000",
      "quote": "exact immutable unit quote",
      "quote_sha256": "...",
      "label": "exact label",
      "span_start": 0
    },
    "witnesses": [{
      "chapter": "NN-part_NN.txt",
      "chapter_sha256": "...",
      "unit_id": "c00s00001",
      "quote": "exact immutable witness quote",
      "quote_sha256": "..."
    }],
    "note": "caretaker rationale",
    "resolution": null
  }]
}
```

`span_start` is zero-based Python-Unicode codepoint offset **within `scope.quote`**, matching `mention_scope`. Each witness is independently checked against its own immutable unit; it is not required to hash to the main quote.

For `status: "resolved"`, `resolution` must be `{ "reason": "..." }`; the original proposal material must be unchanged. The producer appends a resolution history row rather than rewriting/deleting the pending row. Invalid source, a changed proposal under one ID, a duplicate exact mention scope, an inactive/unknown actor, or `narrator` fails closed. Unresolved flags for active actors block the final cast manifest.
