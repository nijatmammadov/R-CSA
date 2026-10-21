You will see a math word problem, the previous steps of a solution, and the current step.

Write ALTERNATIVE versions of the current step: the step a solver would have written if they had made ONE different choice at this point. Do not judge which version is right. Just write the alternatives.

Alternative types:
- logical: use a different operation or reverse a relation (add instead of subtract, divide instead of multiply, "3 times as many" the other way round, more instead of less).
- factual: use a different quantity from the question or from the previous steps in place of one the step uses, or drop an assumption the question does not state.
- temporal: use a different state or moment (before instead of after, remaining instead of total, start instead of end, per period instead of overall).
- quantitative: keep the setup exactly, but redo every calculation of the step yourself, digit by digit. Write the step with your own results. If every result is the same as in the step, write NONE.
- pragmatic: re-read what the question asks. Write the step that computes what the question needs at this point. If the step already does that, write NONE.

Rules:
1. Change ONE choice. Keep the rest of the wording of the step.
2. Compute the alternative's own numbers correctly. Never keep the old result after changing the calculation.
3. The alternative must be plausible: something a careful solver could write.
4. If a type does not apply, write NONE. Never change a number at random.

Example 1
Question: A shop has 40 apples. It sells 12 in the morning and receives 10 more in the afternoon. How many apples does it have at the end of the day?
Previous steps: (none)
Current step: In the morning it sells 12, so 40 - 12 = 28 apples are left.
{"choices": "subtracts the 12 sold from the 40 at the start", "logical": "In the morning it sells 12, so 40 + 12 = 52 apples are left.", "factual": "In the morning it sells 10, so 40 - 10 = 30 apples are left.", "temporal": "NONE", "quantitative": "NONE", "pragmatic": "NONE"}

Example 2
Question: A shop has 40 apples. It sells 12 in the morning and receives 10 more in the afternoon. How many apples does it have at the end of the day?
Previous steps:
  1. In the morning it sells 12, so 40 - 12 = 28 apples are left.
Current step: After the delivery it has 40 + 10 = 50 apples.
{"choices": "adds the 10 delivered to the 40 at the start", "logical": "After the delivery it has 40 - 10 = 30 apples.", "factual": "After the delivery it has 40 + 12 = 52 apples.", "temporal": "After the delivery it has 28 + 10 = 38 apples.", "quantitative": "NONE", "pragmatic": "NONE"}

Now the real step. Reply with one JSON object only, with the keys "choices", "logical", "factual", "temporal", "quantitative", "pragmatic".

Question: <<QUESTION>>
Previous steps:
<<PRIOR>>
Current step: <<STEP>>
