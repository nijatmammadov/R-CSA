You are checking ONE step of a solution to a math word problem.

You are given the question, the previous steps of the solution, and a candidate step.

Decide whether what the candidate step states is correct:
- Recompute every calculation in the step yourself.
- Check that each number taken from the question is used the way the question states it
  (a rate per day vs a total, which person or day it belongs to, "of the original" vs
  "of the new").
- If the step uses a value from a previous step and that value is wrong, the step is wrong.
- A step that does only part of the work is CORRECT if that part is right. Do not lower the
  score because the step does not use all the information yet or does not reach the final
  answer; later steps do that.

Return ONLY this JSON, fields in this order:
{"reasoning": "one sentence", "error_found": "<the specific error, or the word none>", "p_correct": 0.00}

p_correct: 0.90-1.00 if there is no error, 0.00-0.20 if there is an error, 0.40-0.60 only
if the question itself is genuinely ambiguous.
