# BookwriterPro — MCP Server

BookwriterPro ships a **Model Context Protocol (MCP)** server so an AI agent
(Claude Desktop, Claude Code, or any MCP client) can plan and write full-length
books as a tool. It speaks the **stdio** transport and reuses the same
application layer (`bookwriter.server.service.BookService`) and the same on-disk
data directory as the HTTP/web UI — so books you create from the agent show up
in the web app, and vice versa.

## What it does

The server exposes seventeen tools. Every tool returns plain JSON. Use `mock=True`
to run **fully offline with no API key** (deterministic placeholder prose) —
ideal for trying the tools before spending tokens.

| Tool | Purpose |
| --- | --- |
| `list_profiles()` | Quality profiles (premium/balanced/draft): which Claude model runs each stage + per-model prices. |
| `list_books()` | Every book in the shared data dir, with progress. |
| `create_book(premise, chapters=12, words_per_chapter=2000, title="", genre="", profile="balanced", mock=False)` | Plan a book **synchronously** (bible + characters + chapter outline). Returns the new book `id` and the full planned `bible`. |
| `write_book(book_id, only=None)` | Write chapters **synchronously**; resumes from the last unwritten chapter. Returns `chapters_written`, `cost`, `flags`. |
| `write_chapter(book_id, number)` | Write/rewrite a single chapter (sugar for `write_book(only=[number])`). |
| `get_status(book_id)` | `{chapters_total, chapters_written}`. |
| `get_chapter(book_id, number)` | A chapter's prose + metadata + its plan. |
| `get_graph(book_id)` | The continuity graph: characters, locations, items, threads, timeline, rolling synopsis. |
| `get_cost(book_id)` | Token-cost snapshot + human-readable report for the last write run. |
| `get_manuscript(book_id)` | The assembled full book as Markdown. |
| `prepare_kdp(book_id, author_first, author_last, ...)` | Generate KDP listing metadata + build the upload kit (EPUB, cover, listing, checklist) into `<book>/kdp/`. |
| `export_epub(book_id)` | Path to the KDP-ready EPUB (builds the kit if needed). |
| `export_docx(book_id)` | Path to the print-ready 6×9 DOCX interior. |
| `print_spec(book_id, paper="white")` | Paperback print spec: estimated page count, spine width, full-wrap cover dimensions. |
| `estimate_royalties(book_id, list_price, marketplace="US", paper="white")` | Estimated eBook + paperback royalties per sale. |
| `generate_marketing(book_id)` | Back-cover blurbs, A+ modules, author bio, taglines (mock mode inherited from the book). |
| `get_kdp_listing(book_id)` | The copy-paste KDP listing text for an already-prepared kit. |

A typical agent flow:

1. `create_book(premise="...", chapters=12, mock=true)` → grab the returned `id`.
2. `write_book(book_id=id)` → blocks until all chapters are written, returns cost + flags.
3. `get_manuscript(book_id=id)` → the finished book in Markdown.

## Requirements

- Python 3.13 (the project's interpreter).
- The `mcp` package: `pip install mcp`. (The module imports `mcp` lazily, so it
  still imports without it; `python -m bookwriter.mcp_server` will print an
  install hint and exit if `mcp` is missing.)
- For **live** generation (not `mock`): set `ANTHROPIC_API_KEY` and
  `pip install anthropic`. Mock mode needs neither.

## Run it (stdio)

```bash
# from the project root: C:\Users\VR\projects\BookwritterPro
python -m bookwriter.mcp_server
```

The server communicates over stdin/stdout — there is no banner; an MCP client
drives it. To sanity-check it starts, run the command and confirm it does not
exit (Ctrl-C to stop), or wire it into a client using the config below.

### Data directory (shared with the web app)

Books live under `<DATA>/<book_id>/`. The default is
`C:\Users\VR\projects\BookwritterPro\.bookwriter_data`. Override it with the
`BOOKWRITER_DATA_DIR` environment variable. **Point the MCP server and the HTTP
server at the same `BOOKWRITER_DATA_DIR`** (or leave both at the default) so
they share state.

## Claude Desktop configuration

Edit Claude Desktop's config file:

- Windows: `%APPDATA%\Claude\claude_desktop_config.json`
- macOS: `~/Library/Application Support/Claude/claude_desktop_config.json`

Add (or merge) the `bookwriter` entry. This is the exact snippet written for
this machine:

```json
{
  "mcpServers": {
    "bookwriter": {
      "command": "C:\\Users\\VR\\AppData\\Local\\Programs\\Python\\Python313\\python.exe",
      "args": ["-m", "bookwriter.mcp_server"],
      "cwd": "C:\\Users\\VR\\projects\\BookwritterPro",
      "env": {
        "BOOKWRITER_DATA_DIR": "C:\\Users\\VR\\projects\\BookwritterPro\\.bookwriter_data",
        "ANTHROPIC_API_KEY": "sk-ant-...replace-or-omit-for-mock-only..."
      }
    }
  }
}
```

Notes:
- `cwd` must be the project root so `python -m bookwriter.mcp_server` resolves
  the `bookwriter` package.
- Omit `ANTHROPIC_API_KEY` if you only ever call tools with `mock=True`.
- Restart Claude Desktop after editing the config. The `bookwriter` tools then
  appear in the tools menu.

## Claude Code configuration

Claude Code reads MCP servers from `.mcp.json` (project scope) or your user
config. The fastest path is the CLI:

```bash
claude mcp add bookwriter \
  --scope user \
  --env BOOKWRITER_DATA_DIR=C:\\Users\\VR\\projects\\BookwritterPro\\.bookwriter_data \
  -- C:\\Users\\VR\\AppData\\Local\\Programs\\Python\\Python313\\python.exe -m bookwriter.mcp_server
```

Or add it by hand to a project-root `.mcp.json` (same schema as Claude Desktop):

```json
{
  "mcpServers": {
    "bookwriter": {
      "command": "C:\\Users\\VR\\AppData\\Local\\Programs\\Python\\Python313\\python.exe",
      "args": ["-m", "bookwriter.mcp_server"],
      "cwd": "C:\\Users\\VR\\projects\\BookwritterPro",
      "env": {
        "BOOKWRITER_DATA_DIR": "C:\\Users\\VR\\projects\\BookwritterPro\\.bookwriter_data"
      }
    }
  }
}
```

Then run `claude` from the project directory and the tools are available.

## Troubleshooting

- **"The 'mcp' package is required…"** — `pip install mcp` into the same
  interpreter named in `command`.
- **`create_book` returns `{"error": "No credentials for LLM provider '...'; ..."}`** — set
  the key in the `env` block, or call with `mock=True`.
- **Tools don't appear** — verify `command` points at the right Python and `cwd`
  is the project root; check the client's MCP logs.
- **Books from the agent aren't in the web app (or vice versa)** — make sure both
  use the same `BOOKWRITER_DATA_DIR`.
