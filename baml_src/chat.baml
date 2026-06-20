// Chat functions for the dejavu demo.
//
// Message assembly order (stable prefix first, changing tail last):
//   system                          — static every turn
//   document (contract)             — static across session
//   history[0..n-1]                 — grows 2 entries per turn
//   question                        — always fresh (must sit BELOW any cache breakpoint)
//
// ChatUncached: no cache_control metadata anywhere.
// ChatCached:   rolling ephemeral marker on the last history message so the whole
//               prefix (system + contract + prior turns) stays cached; the fresh
//               question remains below the breakpoint.

class Message {
  role    string
  content string
}

// ── No-cache variant ────────────────────────────────────────────────────────

function ChatUncached(document: string, history: Message[], question: string) -> string {
  client Sonnet46
  prompt #"
    {{ _.role("system") }}
    You are a helpful assistant that answers questions about a contract document.
    Answer concisely and accurately based only on the document provided.

    {{ _.role("user") }}
    Contract document:
    {{ document }}

    {% for msg in history %}
    {{ _.role(msg.role) }}
    {{ msg.content }}
    {% endfor %}

    {{ _.role("user") }}
    {{ question }}
  "#
}

// ── Rolling-cache variant ────────────────────────────────────────────────────

function ChatCached(document: string, history: Message[], question: string) -> string {
  client Sonnet46
  prompt #"
    {{ _.role("system") }}
    You are a helpful assistant that answers questions about a contract document.
    Answer concisely and accurately based only on the document provided.

    {{ _.role("user", cache_control={"type": "ephemeral"}) }}
    Contract document:
    {{ document }}

    {% if history %}
    {% for msg in history %}
    {% if loop.last %}
    {{ _.role(msg.role, cache_control={"type": "ephemeral"}) }}
    {% else %}
    {{ _.role(msg.role) }}
    {% endif %}
    {{ msg.content }}
    {% endfor %}
    {% endif %}

    {{ _.role("user") }}
    {{ question }}
  "#
}
