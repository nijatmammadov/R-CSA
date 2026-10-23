You will see a yes/no question, the previous steps of a reasoning chain, and the current step, written as one claim.

Write COUNTERFACTUAL versions of the current step. A counterfactual is the same claim with exactly ONE thing changed, as described below. Each counterfactual is a claim a solver could really have written at this point: plausible, not absurd. Do not judge which version is right.

The five kinds:

- factual: replace ONE entity, property or category in the claim with a different, plausible one (another person, place, object, material, group). Keep everything else.
- temporal: change ONE time reference: a year, date, century, period, duration, or the order of two events (before/after). If the claim has no time reference, write NONE.
- quantitative: change ONE number or amount that is not a time reference to a different plausible value (a count, size, distance, weight, percentage). If the claim has no such number or amount, write NONE.
- logical: reverse the claim in ONE place: negate it, reverse a comparison (greater/smaller, more/fewer, before/after a threshold), or change a quantifier (all/some/none, always/sometimes).
- causal: change ONE relation between two things: cause into effect, cause into mere correlation, "so" into "but", a requirement into an option. If the claim states no relation, write NONE.

Rules for every counterfactual:
1. One sentence of about the same length and form as the current step. No explanation, no restating the question.
2. Change exactly one thing and keep the rest word for word.
3. If a kind does not apply, write NONE. Never make a change no solver would plausibly write.

Example
Question: Could a person walk across the Golden Gate Bridge in under an hour?
Previous steps:
  1. The Golden Gate Bridge is about 2.7 kilometers long.
Current step: A person walks about 5 kilometers per hour.
{"factual": "A person swims about 5 kilometers per hour.", "temporal": "NONE", "quantitative": "A person walks about 1 kilometer per hour.", "logical": "A person cannot walk 5 kilometers per hour.", "causal": "NONE"}

Example
Question: Was the telephone invented before the light bulb?
Previous steps:
  1. Alexander Graham Bell patented the telephone in 1876.
Current step: Thomas Edison patented his light bulb in 1880.
{"factual": "Nikola Tesla patented his light bulb in 1880.", "temporal": "Thomas Edison patented his light bulb in 1870.", "quantitative": "NONE", "logical": "Thomas Edison never patented a light bulb.", "causal": "NONE"}

Example
Question: Would a vegetarian eat a traditional Caesar salad?
Previous steps:
  1. Traditional Caesar salad dressing contains anchovies.
  2. Anchovies are fish.
Current step: Vegetarians do not eat fish, so a vegetarian would not eat a traditional Caesar salad.
{"factual": "Vegetarians do not eat eggs, so a vegetarian would not eat a traditional Caesar salad.", "temporal": "NONE", "quantitative": "NONE", "logical": "Vegetarians eat fish, so a vegetarian would eat a traditional Caesar salad.", "causal": "Vegetarians do not eat fish, but a vegetarian would still eat a traditional Caesar salad."}

Now the real step. Reply with one JSON object only, with the keys "factual", "temporal", "quantitative", "logical", "causal".

Question: <<QUESTION>>
Previous steps:
<<PRIOR>>
Current step: <<STEP>>
