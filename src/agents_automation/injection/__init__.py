"""Layered screening of a finished step list for planted instructions. (Documents and chat
messages get the Layer 1 patterns only, at ingest and load time.)

Layer 1  high-precision patterns (authority stamps, "skip review" ...)  -> held, no judge call
Layer 2  broad prefilter: does the text use any authority vocabulary?    -> if not, delivered
Layer 3  LLM judge returns {planted, claims}                             -> held or delivered

Every judge failure (transport, timeout, unparseable or inconsistent answer) fails OPEN: the
record is delivered but flagged, so a broken judge can never silently block real work, and
can never silently pass it either.
"""
