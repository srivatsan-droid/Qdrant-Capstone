# Test Queries

Five natural-language queries used throughout the project (distance metrics,
HNSW vs exact, and the from-scratch IVF index). Each is aimed at a different
20 Newsgroups category so the expected "correct neighborhood" is easy to
sanity-check by eye.

1. "a question about a graphics card driver"
   — expected neighborhood: `comp.graphics` / `comp.sys.ibm.pc.hardware`

2. "discussion about God and religious faith"
   — expected neighborhood: `soc.religion.christian` / `alt.atheism`

3. "advice on treating a medical illness"
   — expected neighborhood: `sci.med`

4. "opinions on strict gun control laws"
   — expected neighborhood: `talk.politics.guns`

5. "technical details about a space shuttle mission"
   — expected neighborhood: `sci.space`

`embed.py` parses these five numbered, quoted lines directly out of this file,
so keep the `N. "..."` format if you edit them.
