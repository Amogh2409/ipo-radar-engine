---
name: Bug report
about: Something produced a wrong number or crashed
labels: bug
---

**What happened**

**What you expected**

**Reproduce**
Symbol, timestamp, and the command you ran. If the engine stored it, the
relevant row helps enormously:

```sql
SELECT * FROM verdicts WHERE symbol='XXX' ORDER BY ts DESC LIMIT 1;
```

**Environment**
Python version, OS, and whether Ollama was running (`--no-llm` or not).
