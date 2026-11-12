# Evaluation protocol — one page for the evaluators

**Study:** expert assessment of automatically generated first replies and of the
historical tickets retrieved to support them. Part of an IEEE Access revision on
retrieval-augmented generation for IT support.

**What you will do.** For each ticket you see two candidate first replies, labelled
A and B. The order is random and the system that produced each reply is not
revealed. You will later see, for the same tickets, two sets of retrieved
historical tickets, again as A and B.

**Block 1 — first replies.** Read the ticket and both replies.
1. Rate each reply independently: **correct procedure / partially correct /
   incorrect / cannot judge**.
2. Mark, for each reply, whether it **contains a wrong or unsafe step** and
   whether it **omits a required step**. These are independent of the overall
   rating: a reply can be partially correct and still contain an unsafe step.
3. Choose: **A better / B better / equal / both incorrect / cannot judge**.
   If both are wrong, choose “both incorrect” even if one is less wrong.

**Block 2 — retrieved historical tickets.** Same procedure for the two retrieved
sets: rate whether each set supports the correct procedure, mark a **misleading or
irrelevant ticket** and a **missing ticket needed for the procedure**, then choose
the better set.

**What “correct procedure” means.** Would an agent who follows this reply (or this
retrieved set) resolve the ticket the way your team would, without an incorrect or
unsafe step? Judge the procedure, not the writing style or length.

**Time.** About 1.5 minutes per ticket. A 20-ticket calibration session takes
about one hour including instructions. The full sample is about 100 tickets,
split into two or three sessions.

**Blinding.** Do not try to guess which system produced which reply. Do not
discuss individual tickets with the other evaluator until both of you have
finished; agreement between independent ratings is part of the result.

**Data.** The tickets are confidential. The application runs on the institute’s
server; only anonymised tickets are shown; ratings are stored without ticket text
in any external service. No files need to be downloaded and nothing is installed
on your machine.

**If something is unclear.** Use “cannot judge” only when the ticket genuinely does
not allow a decision. Use the comment box for anything the rubric does not capture
— that is often the most useful information.

**Contact.** Domen Jeršek, Jožef Stefan Institute — for access, scheduling or
technical issues with the application.
