"""Plan-then-execute, shared by every role.

route -> plan -> seal -> authorize happen in the *trusted zone*, from the request envelope
only. Untrusted content (conversation text, stored records, documents, chat messages) is read
later, by the executor's tools and the roles, and can only fill slot *values* of the sealed
plan, never change its structure.
"""
