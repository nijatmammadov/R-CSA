You will see a math word problem, the previous steps of a solution, and the current step. Each step is one sentence of the form

  <quantity> = <expression> = <value>

Write ALTERNATIVE versions of the current step: the step a solver would have written if they had made ONE different choice at this point. Do not judge which version is right. Just write the alternatives.

Alternative types:
- logical: a different operation, or a relation reversed (add instead of subtract, divide instead of multiply, "3 times as many" the other way round).
- factual: a different quantity from the question or from a previous step in place of one this step uses.
- temporal: a different state or moment (before instead of after, remaining instead of total, start instead of end, per period instead of overall).
- quantitative: keep the expression exactly, but redo the arithmetic yourself, digit by digit, and write your own value. If your value is the same as the step's, write NONE.
- pragmatic: re-read what the question asks. Write the step that computes what the question needs at this point. If the step already does that, write NONE.

Rules:
1. Keep the SAME quantity name on the left of the "=" as the current step has. Change only the expression and the value.
2. Change ONE choice per alternative.
3. Compute the alternative's own value correctly: the value must follow from its own expression.
4. Each alternative is one sentence in the same form. Never restate the question, and never add words that are not a quantity, an expression or a value.
5. If a type does not apply, write NONE. Never change a number at random.

Example
Question: A shop has 40 apples. It sells 12 in the morning and receives 10 more in the afternoon. How many apples does it have at the end of the day?
Previous steps: (none)
Current step: apples after selling = 40 - 12 = 28
{"logical": "apples after selling = 40 + 12 = 52", "factual": "apples after selling = 40 - 10 = 30", "temporal": "NONE", "quantitative": "NONE", "pragmatic": "NONE"}

Example
Question: A shop has 40 apples. It sells 12 in the morning and receives 10 more in the afternoon. How many apples does it have at the end of the day?
Previous steps:
  1. apples after selling = 40 - 12 = 28
Current step: apples after delivery = 40 + 10 = 50
{"logical": "apples after delivery = 40 - 10 = 30", "factual": "apples after delivery = 28 + 10 = 38", "temporal": "NONE", "quantitative": "NONE", "pragmatic": "NONE"}

Now the real step. Reply with one JSON object only, with the keys "logical", "factual", "temporal", "quantitative", "pragmatic".

Question: <<QUESTION>>
Previous steps:
<<PRIOR>>
Current step: <<STEP>>
