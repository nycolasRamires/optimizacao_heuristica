# Project Rules & Agent Guidelines

## 1. Environment & Execution
- **Python Version:** ALWAYS use **Python 3.12** for all command executions and script calls.
  - Active virtual environment: `.venv/bin/python3.12`
  - When running Python scripts or tests, use: `.venv/bin/python3.12 <script.py>` or ensure `PATH=.venv/bin:$PATH`.
- **Shared Lab Machine:** Always maintain environment packages and changes strictly within `.venv/`.

## 2. AI Documentation & Project Tracking (`ai/` Directory)
All agent-maintained documentation and tracking files MUST be placed inside the `ai/` directory:
- **[`ai/Goals.md`](file:///home/nycolas/m/ai/Goals.md) (Goals & Progress):** Concise, clear checklist with hierarchical sub-goals.
- **[`ai/Explanations.md`](file:///home/nycolas/m/ai/Explanations.md) (Technical Explanations & Rationale):** Technical rationales, equations, and profiling for 3D simulation and RL.
- **[`ai/Decisions.md`](file:///home/nycolas/m/ai/Decisions.md) (Engineering Decisions Log):**
  - **MINIMAL & CONCISE:** Record ONLY major architectural pivots, trade-offs, and critical design choices. Avoid logging trivial parameter tweaks or minor fixes.

## 3. Core Project Context (Branch `main` - 3D Quadrotor Navigation)
- **Objective:** 3D Quadrotor Autonomous Navigation & Obstacle Avoidance via Reinforcement Learning and Genetic Algorithms.
- **Key Modules:**
  - [`uav_env.py`](file:///home/nycolas/m/uav_env.py): 3D PyBullet drone environment (`Drone3DEnv`) based on `gym-pybullet-drones`.
  - [`train.py`](file:///home/nycolas/m/train.py): Stable-Baselines3 SAC training loop with `MultiInputPolicy`.
  - [`enjoy.py`](file:///home/nycolas/m/enjoy.py): 3D third-person video recorder and policy visualizer.
  - [`ai/`](file:///home/nycolas/m/ai): Centralized directory for goals, technical explanations, and concise decisions.

## 4. Operational Safeguards
- **Zero-Modification Invariant:** NEVER modify, edit, or delete codebase files or scripts while an active background training/evolution process is running on the GPU.
- **Process Verification:** Before editing core files, verify active training status (`ps aux | grep python`).
- **Deterministic Evaluation Standard:** In SAC and actor-critic setups, always benchmark and rank individuals using deterministic mode (`deterministic=True`) over $\ge 100$ seeded episodes to eliminate evaluation variance.

## 5. Obsidian Vault & Markdown Rules
You are integrated with an Obsidian Vault via the filesystem MCP. When generating or modifying `.md` documentation, you must adhere to the following strict rules to ensure a pristine, interconnected knowledge base:

1. **Read Before Write:** Always use the filesystem tools to search the target directory for existing files or similar concepts before creating a new note. Avoid duplicating knowledge.
2. **YAML Frontmatter:** Every new markdown file MUST begin with valid YAML frontmatter. Include `date`, `tags`, and `aliases` (if relevant). Example:
   ---
   date: YYYY-MM-DD
   tags: [agent, generated, code-docs]
   aliases: []
   ---
3. **Bidirectional Links (WikiLinks):** Use Obsidian's native `[[WikiLink]]` syntax to reference internal files, concepts, or related architectural components. Do NOT use standard markdown links `[text](file.md)` for internal files.
4. **File Naming:** Create concise, descriptive file names using PascalCase. Never use spaces in file names.
5. **Non-Destructive Editing:** When modifying existing files, read the current content first. Preserve existing frontmatter, tags, and internal links. Append new data logically under relevant headers.
6. **Code Blocks & Formatting:** Always specify the language in fenced code blocks (e.g., ```python). Use clear hierarchical headings (`##`, `###`) to structure long technical documents.
7. **Bottom Tagging:** Include a "Related" section at the bottom of the document with relevant inline tags (e.g., `#architecture`, `#cli`) and a list of related `[[Links]]` to improve the Obsidian Graph View.
8. **Knowledge Traceability (Problems, Investigations, Decisions):** Separate files by folder instead of prefixing file names:
   - **Problems (`ai/problems/[Name].md`):** Document bugs or requirements. Tag with `#type/problem` and `#status/open` or `#status/resolved`.
   - **Investigations (`ai/investigations/[Name].md`):** Document solution explorations. Tag with `#type/investigation`. The first line must link back: `> Addresses: [[problems/[Name]]]`.
   - **Decisions (ADRs, `ai/decisions/[Name].md`):** Document architectural decisions. Tag with `#type/decision`. The first line must link: `> Resolves: [[investigations/[Name]]]` and `> Related to: [[problems/[Name]]]`.
   - **Architecture (`ai/architecture/[Name].md`):** System and perception architecture specifications.
9. **Tagging Hierarchy:** Always include hierarchical tags in the YAML frontmatter to enable filtering in Obsidian (e.g., `tags: [type/decision, domain/simulation, status/resolved]`).
10. **Update States:** When a decision is made or a problem is solved, edit the original `Problem` and `Investigation` files to update their frontmatter tags to `#status/resolved` and append a link to the final `[[decisions/[Name]]]` at the top of the file.

## 6. Technical Writing & Communication Style (ASD-STE100)
Apply these rules to all documentation, codebase comments, commit messages, and agent communications:

### Core Rules
- Never use a long word where a short word works.
- Always remove unnecessary words.
- Never use passive voice where active voice works.

### Word Choice & Vocabulary
- **Consistent Terms:** Use one name for one thing. Do not alternate synonyms for the same action (e.g., do not rotate *check / verify / validate / confirm*; use "make sure" or "examine").
- **Use Short Common Words:**
  - *start* (not begin, commence, initiate)
  - *use* (not utilize, leverage)
  - *help* (not facilitate)
  - *make sure* (not ensure, verify)
  - *do* (not perform, conduct)
  - *give* or *supply* (not provide)
  - *before* (not prior to)
  - *after* (not subsequent to)
  - *about* (not regarding, concerning)
  - *get* (not obtain, acquire)
  - *show* (not demonstrate)
  - *also* (not additionally, furthermore, moreover)
- **One Meaning Per Word:** Give each word one meaning. For example, "fall" means move down, not decrease.
- **No Marketing Adjectives:** Do not use words such as *seamless, robust, powerful, cutting-edge, effortless, world-class, next-generation, revolutionary*.
- **American Spelling:** Use standard American spelling.

### Verbs & Voice
- **Active Voice:** Write "the parser reads the file", not "the file is read by the parser". Use passive voice only when the actor is unknown or irrelevant.
- **Participles as Adjectives:** Past participles used as adjectives are permitted (e.g., "the valve is closed", "the field is required").
- **Simple Tenses Only:** Use infinitive, imperative, simple present, simple past, and simple future. Do not use present perfect (write "we received the report", never "we have received the report").
- **No Stacked Auxiliaries:** Avoid stacked auxiliaries (write "this improves X", not "it is important to note that this may help to improve").
- **Use Verbs for Actions:** Write "analyze the log", not "perform an analysis of the log".
- **No "-ing" Main Verbs:** Do not use "-ing" main verbs when a simple tense works.
- **No Phrasal Verbs:** Do not use phrasal verbs (e.g., avoid *spin up, dive into, kick off, roll out*).

### Sentence Construction
- **One Instruction per Sentence:** Keep to one instruction per sentence unless two actions occur at the same time.
- **Sentence Length Limits:** Maximum 20 words for instructions; maximum 25 words for descriptions.
- **Conditionals:** When a condition comes before its command, separate them with a comma (e.g., "If the test fails, read the log.").
- **No Compression Shortcuts:** Do not drop words to compress sentences (write "Remove the bolts from the panel", never "Remove bolts from panel"). Do not use contractions.
- **Articles and Demonstratives:** Use an article (*a, an, the*) or demonstrative (*this, these*) before a noun where appropriate. Do not add articles to abstract general concepts. In a series of items, an article before the first noun is sufficient.
- **Plain Connectors:** Connect related sentences with plain connectors (*then, but, thus, as a result*).

### Nouns & Abbreviations
- **Multi-Word Nouns:** Keep multi-word nouns to at most three words. Unpack or hyphenate long noun compounds (e.g., write "the handler that sets task-queue priority" instead of "the agent task queue priority handler").
- **Abbreviations:** Define an abbreviation on first use, then use the abbreviation.

### Punctuation & Paragraph Structure
- **No Semicolons:** Do not use semicolons. Write two sentences instead.
- **Paragraph Bounds:** One topic per paragraph; maximum six sentences per paragraph.
- **Procedures and Steps:** Use a numbered vertical list with one action per item in the imperative form. Put conditions before commands.
- **List Labels:** Short labels without articles are permitted for list items (e.g., changelogs or feature bullets).
- **Safety Notices:** Place safety notices directly before the affected step:
  - `WARNING`: Risk of injury.
  - `CAUTION`: Risk of damage.
  - `NOTE`: Information only (never an instruction).
  - Format: State the command or condition first, then state the risk.


