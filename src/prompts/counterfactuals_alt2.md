You will see a math word problem, the previous steps of a solution, and the current step. Each step is one sentence of the form

  <quantity> = <expression> = <value>

Write COUNTERFACTUAL versions of the current step. A counterfactual is the same step with exactly ONE dimension changed, as defined below. Each counterfactual is a step a solver could really have written at this point. Do not judge which version is right.

The five dimensions (from Feng 2025, applied to one reasoning step):

- logical: "introduce a logical inconsistency or an invalid relation". Change ONLY the operation or the direction of the relation: add instead of subtract, divide instead of multiply, "3 times as many" read the other way round. Keep every number of the expression.
- factual: "alter a key entity, relationship or attribute". Replace ONE number of the expression with a DIFFERENT quantity stated in the question (a different item, person or attribute). Keep the operation. Do not use a duration here, and do not use the result of an earlier step.
- temporal: "modify time-related information: dates, durations or temporal sequences". Replace ONE number of the expression that the question attaches to a unit of time (minutes, hours, days, weeks, months, years, a clock time) with a different time-related number from the question, or with the same duration counted from another moment (before/after, start/end). Keep the operation. If the expression uses no time-related number, write NONE.
- quantitative: "perturb numerical values or measurements". Keep the expression exactly. Redo its arithmetic yourself, digit by digit, and write YOUR value. If your value is the same as the step's value, write NONE.
- causal: "change which earlier result the step builds on". Replace ONE number of the expression with the result (value) of a DIFFERENT earlier step. Keep the operation. If there are no earlier steps, or the step already uses the only one, write NONE.

Rules for every counterfactual:
1. Keep the SAME quantity name on the left of the "=" as the current step. Change only the one dimension of that type.
2. Except for quantitative, compute the counterfactual's own value correctly from its own expression.
3. One sentence in the same form. No explanation, no restating the question.
4. If a type does not apply, write NONE. Never change a number at random.

Example
Question: Ana works 6 hours a day for 5 days and is paid $12 an hour. Her bus pass costs $15 a week. How much does she keep in a week?
Previous steps:
  1. hours worked = 6 * 5 = 30
Current step: pay = 30 * 12 = 360
{"logical": "pay = 30 / 12 = 2.5", "factual": "pay = 30 * 15 = 450", "temporal": "pay = 6 * 12 = 72", "quantitative": "NONE", "causal": "NONE"}

Example
Question: Ana works 6 hours a day for 5 days and is paid $12 an hour. Her bus pass costs $15 a week. How much does she keep in a week?
Previous steps:
  1. hours worked = 6 * 5 = 30
  2. pay = 30 * 12 = 360
Current step: money kept = 360 - 15 = 345
{"logical": "money kept = 360 + 15 = 375", "factual": "money kept = 360 - 12 = 348", "temporal": "NONE", "quantitative": "NONE", "causal": "money kept = 30 - 15 = 15"}

Now the real step. Reply with one JSON object only, with the keys "logical", "factual", "temporal", "quantitative", "causal".

Question: <<QUESTION>>
Previous steps:
<<PRIOR>>
Current step: <<STEP>>
