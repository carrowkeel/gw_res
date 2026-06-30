Factual basis/reliability:
1. When uncertain, always ask the user for input. Never assume anything about code you have not seen.
2. The user will prompt you to confirm changes before coding. Do not write code without being asked to.
3. Do not make any changes to code that were not explicitly requested.
Stylistic Code Guidelines:
2. **Character Set:** Never use non-ASCII characters anywhere in code, including comments and docstrings.
3. **Typing:** Do not use Python typing or type hints in function definitions.
4. **Quotations:** Use single quotation marks (' ') where possible, instead of double quotes (" ").
5. **Comments:**
- Do not add inline comments next to code.
- Do not write section comments or blocks of comments that describe code sections.
- Do not write comments that refer to changes, other messages, or the process (e.g., avoid comments such as "# Changed from the previous version" or that answer questions from our conversation).
6. **Variable Names:** Use descriptive variable names that make code self-explanatory and reduce the need for comments. JavaScript variable names should be underscore-delimited, with only JavaScript function names being camelCase. No abbreviations or cryptic shortened words for variable names, those are hard to read and create inconsistencies in shortening styles across the code.
7. **Docstrings:** Write good docstrings for important functions when necessary, rather than comments.
8. **General:** Follow these stylistic rules in all future code writing.

# Intent Graph
Directed acyclic graph of crystallized human intent replacing linear LLM conversation history.
Nodes are arrays of content strings indexed by resolution level (default: 0=title, 1=summary, 2=description, 3=full).
Resolution scheme is defined in `meta.json` per graph; each level has a name and target_tokens (null for unbounded full level).
Edges are stored as a flat array of `{from, to, weight}` objects in `edges.json`; undirected, weight 0-1, weight 0 removes the edge.
Edge weight maps to resolution level via `floor(weight * (level_count - 1))`, controlling how much detail to load from neighbors.
Node solidity emerges from topology: degree centrality (sum of edge weights / live node count - 1) determines foundation status.
Nodes grow then split (sibling: edges duplicated to new node; parent-child: child gets single weight-1.0 edge to parent).
Merged nodes are cleared to empty arrays `[]` to preserve indices; `meta.json` tracks `node_count` as a monotonic counter.
Storage: `.intent/{graph_id}/meta.json`, `edges.json`, `nodes/{index}.json`; all reads via `getFile`, writes via `writeFile`, atomic commits via `putFiles`.
Terminal state: when the graph fully captures human intent, no further interaction is necessary.
