You will see a math word problem, the previous steps of a solution, and the current step. Each step is one sentence of the form

  <quantity> = <expression> = <value>

Write COUNTERFACTUAL versions of the current step. A counterfactual is the same step with exactly ONE thing changed, as described below. Each counterfactual is a step a solver could really have written at this point. Do not judge which version is right.

The six kinds:

- logical: change ONLY the operation or the direction of the relation: add instead of subtract, divide instead of multiply, "3 times as many" read the other way round. Keep every number of the expression.
- factual: replace ONE number of the expression with a DIFFERENT quantity stated in the question (a different item, person or attribute). Keep the operation.
- temporal: replace ONE number of the expression that the question attaches to a unit of time (minutes, hours, days, weeks, months, years, a clock time) with a different time-related number from the question, or with the same duration counted from another moment (before/after, start/end). If the expression uses no time-related number, write NONE.
- quantitative: keep the expression exactly. Redo its arithmetic yourself, digit by digit, and write YOUR value. If your value is the same as the step's value, write NONE.
- causal: replace ONE number of the expression with the result (value) of a DIFFERENT earlier step. If there are no earlier steps, or the step already uses the only one, write NONE.
- reading: re-read the question for what this step's quantity refers to: per what unit, of whom, which part of a whole, whether something is included or left out, "times as many" as what. Write the step a solver would write under the OTHER reading: swap one number for another quantity of the question, or add one quantity the step left out, or leave out one it included. If there is no other reasonable reading, write NONE.

Rules for every counterfactual:
1. Keep the SAME quantity name on the left of the "=" as the current step.
2. Except for quantitative, compute the counterfactual's own value correctly from its own expression.
3. One sentence in the same form. No explanation, no restating the question.
4. If a kind does not apply, write NONE. Never change a number at random.

Example
Question: Ana works 6 hours a day for 5 days and is paid $12 an hour. Her bus pass costs $15 a week. How much does she keep in a week?
Previous steps:
  1. hours worked = 6 * 5 = 30
Current step: pay = 30 * 12 = 360
{"logical": "pay = 30 / 12 = 2.5", "factual": "pay = 30 * 15 = 450", "temporal": "pay = 6 * 12 = 72", "quantitative": "NONE", "causal": "NONE", "reading": "NONE"}

Example
Question: Ana works 6 hours a day for 5 days and is paid $12 an hour. Her bus pass costs $15 a week. How much does she keep in a week?
Previous steps:
  1. hours worked = 6 * 5 = 30
  2. pay = 30 * 12 = 360
Current step: money kept = 360 - 15 = 345
{"logical": "money kept = 360 + 15 = 375", "factual": "money kept = 360 - 12 = 348", "temporal": "NONE", "quantitative": "NONE", "causal": "money kept = 30 - 15 = 15", "reading": "money kept = 360 - 15 * 5 = 285"}

Example
Question: A box has 12 red pens and 3 times as many blue pens as red pens. It also has 5 green pens. How many pens are in the box?
Previous steps:
  1. blue pens = 12 * 3 = 36
Current step: pens in the box = 12 + 36 = 48
{"logical": "pens in the box = 36 - 12 = 24", "factual": "pens in the box = 12 + 5 = 17", "temporal": "NONE", "quantitative": "NONE", "causal": "NONE", "reading": "pens in the box = 12 + 36 + 5 = 53"}

Now the real step. Reply with one JSON object only, with the keys "logical", "factual", "temporal", "quantitative", "causal", "reading".

Question: <<QUESTION>>
Previous steps:
<<PRIOR>>
Current step: <<STEP>>
