"""Zero-shot prompt experiments (2026-07-18).

Chat-style zero-shot variants:

  - ``PureZeroShot``            (``--prompt zeroshot``)    — plain instruction only.
  - ``LexiconGuided``         (``--prompt lexicon``) — instruction + domain lexicon.
  - ``LogicalFormGuided``       (``--prompt lf``)             — instruction + logical-form guidance.
  - ``LexiconLogicalFormGuided`` (``--prompt lexicon-lf``)    — instruction + lexicon + logical-form
    guidance, both blocks together. This is the fourth cell of the 2x2 factorial
    {no lexicon, lexicon} x {no lf, lf} -- pairs with the three above to let a
    lexicon:lf interaction effect be estimated (does combining the two help
    more/less than the sum of their individual effects), not just their two
    main effects.

Each variant differs ONLY in its system prompt constant below — edit the
``*_SYSTEM`` strings to tune the experiment; the user turn is identical
(the bare question) across all variants so the comparison stays clean.
"""
from __future__ import annotations

from ..data_loader import Sample
from .base import ChatMessage, PromptStrategy


# ---------------------------------------------------------------------------
# System prompts.
#
# Controlled-experiment design: all three variants share the SAME base header
# and footer (byte-identical), and differ ONLY by the block inserted between
# them (nothing / lexicon / logical form). Edit _BASE_HEADER or _FOOTER to
# change all variants at once; edit a *_BLOCK to change one treatment.
# ---------------------------------------------------------------------------
_BASE_HEADER = """\
### You are a SQL expert for a single-table process mining event log.
### You translate a natural-language question into SQL.
### This event log database describes a travel expense reimbursement process at a university.
# SCHEMA:
# event_log(id INT PRIMARY KEY, activity TEXT, timestamp DATETIME, resource TEXT, cost NUMERIC, idcase TEXT)

"""

_FOOTER = """\
### Reply with ONLY the SQL query, nothing else.
"""

_LEXICON_BLOCK = """\
## LEXICON FALLBACK

Use this fallback only when the answer entity is not already unambiguously
identified by the question's syntax and explicit wording.

First identify the answer entity: the thing the question asks to return,
count, list, compare, or aggregate. Determine the syntactic head noun of
that answer entity and map it to Event, Case, Activity, or Resource.

IMPORTANT PRIORITY RULES
1. An explicit answer entity takes priority over trigger words appearing in
   modifiers, examples, quoted values, or process-context phrases.
2. A specific compound phrase takes priority over a bare trigger word.
   Example: "process instance" → Case, even though bare "instance" → Event.
3. A named activity or resource value is not automatically the answer entity.
   It may instead be a filter value for the corresponding column.
4. Process-context phrases do not create an activity or resource filter unless
   the question explicitly identifies a concrete value or asks for that
   entity.
5. Do not infer an entity solely from a trigger word when the surrounding
   syntax clearly assigns that word another role.

Event
  Use when the question asks about individual rows in the event log:
  a single occurrence or execution of an activity at a point in time.

  Trigger words:
  event, occurrence, performance, instance, record, moment, row, line,
  tuple, intervention.

  The word "action" maps here when it refers to a single execution or
  occurrence, not to an activity type (see Activity below).

  Compound-phrase rule:
  "process instance" → Case.
  Bare "instance" → Event only when it refers to an individual event-log
  occurrence rather than a process instance.

  A named activity value does not change the answer entity:
  "How many events occurred for the activity 'Start trip'?"
  → answer entity = Event
  → filter = activity = 'Start trip'

  Compiles to:
    The query targets event_log rows directly.

    SELECT <event-level column(s)>
    FROM event_log
    [WHERE ...]

  Do NOT GROUP BY idcase, activity, or resource unless the question explicitly
  asks for a per-case, per-activity, or per-resource breakdown.

Case
  Use when the question asks about process instances: groups of events
  sharing the same idcase value.

  Trigger words:
  case, process instance, declaration, travel declaration, permit, trip,
  process.

  "declaration(s)" and "permit(s)" are Case triggers only when they function
  as the answer entity or refer to process instances. When they occur inside
  a process-context phrase, they do not automatically create a filter.

  "processing of declarations" is a process-context expression when it
  appears in constructions such as:
    "during the processing of declarations"
    "in the processing of declarations"
    "when processing declarations"
    "throughout the declaration process"

  In these constructions, the phrase describes the process being studied.
  It does NOT mean:
    activity = 'declaration'
  and does NOT create an activity LIKE filter.

  "trip" is a weak Case trigger. It is frequently part of activity-name
  paraphrases such as:
    "start of a trip"
    "end of a trip"
    "beginning of the trip"
    "finalization of a trip"

  These refer to the Activity values:
    'Start trip'
    'End trip'

  Therefore:
    start/end/beginning/finalization + trip
      → Activity context or activity filter, not Case

  Only a bare "trip" with no such activity-oriented wording may map to Case.

  Compiles to:
    The query targets the idcase column.

    SELECT [DISTINCT] idcase
    FROM event_log
    [WHERE ...]
    [GROUP BY idcase [HAVING ...]]

  Use GROUP BY idcase when the condition depends on a property of the whole
  case, such as a count, minimum, maximum, or sum over that case's events.

Activity
  Use when the question asks about activity types: distinct values of the
  activity column, rather than individual executions of those activities.

  Trigger words:
    activity, activities, task, action-type.

  The word "action" maps here when it refers to an activity type, category,
  or kind of action rather than a single execution.

  "different" or "distinct" before a plural activity-like noun is a strong
  signal for Activity:
    "how many different activities"
    "how many distinct actions"

  A named activity value inside a question is NOT automatically the answer
  entity. It may instead be a filter on the activity column.

  Examples:
    "How many events occurred for 'Start trip'?"
      → answer entity = Event
      → filter = activity = 'Start trip'

    "Which activities include 'Start trip'?"
      → answer entity = Activity

    "How many different activities were performed?"
      → answer entity = Activity

  Activity values may be multi-word phrases and need not contain the word
  "activity", for example:
    'Start trip'
    'End trip'
    'Send reminder'
    'Rejected by supervisor'

  Compiles to:
    The query targets the activity column.

    SELECT [DISTINCT] activity
    FROM event_log
    [WHERE ...]
    [GROUP BY activity [HAVING ...]]

  Use GROUP BY activity when the question asks for a metric broken down by
  activity type, such as a count per activity.

Resource
  Use when the question asks about people or roles performing events.

  Trigger words:
    resource, employee, person, people, collaborator.

  Role names used as the answer entity also map to Resource:
    supervisor, director, administration, administrator, pre-approver,
    approver, budget owner, budget holder.

  A role name appearing inside a named activity value is not automatically
  a Resource answer entity.

  Examples:
    "Which employees performed the events?"
      → answer entity = Resource

    "How many events were performed by supervisors?"
      → answer entity = Resource or resource filter, depending on the
        requested output

    "How many 'Rejected by supervisor' activities occurred?"
      → answer entity = Activity
      → 'supervisor' is part of the activity value, not a Resource entity

  Compiles to:
    The query targets the resource column.

    SELECT [DISTINCT] resource
    FROM event_log
    [WHERE ...]
    [GROUP BY resource [HAVING ...]]

  Use GROUP BY resource when the question asks for a metric broken down by
  resource, such as a count per employee.


## PROCESS-CONTEXT RULE

A Case- or Activity-related noun inside a recognized process-context phrase
does not automatically become the answer entity and does not automatically
create a SQL filter.

Treat the phrase as context when it describes the process or dataset being
studied, especially in constructions such as:

  "during the processing of declarations"
  "in the processing of declarations"
  "during travel declaration processing"
  "in the travel declaration process"
  "within the declaration process"
  "when processing declarations"
  "throughout the process"
  "across the event log"
  "for declaration processing actions"
  "to process the declarations"

These phrases describe the process context, not a literal activity value.

Therefore:
  Do NOT translate "declaration", "processing", or similar process-context
  wording into:
    activity = ...
  activity LIKE ...
  or another keyword-based activity filter,

unless the question explicitly identifies a concrete activity value or
explicitly asks for activities matching that value.

The process-context rule must not override an explicit entity or value filter.

For example:
  "How many events occurred during the processing of declarations?"
    → answer entity = Event
    → "processing of declarations" = process context
    → no activity filter

  "How many events occurred for the activity 'Process declaration'?"
    → answer entity = Event
    → explicit activity filter
    → activity = 'Process declaration'


## FINAL RESOLUTION ORDER

When multiple interpretations are possible, prefer:

  1. Explicit answer entity
  2. Explicit named/quoted value or filter
  3. Specific compound phrase
  4. Syntactic head noun
  5. Recognized process-context construction
  6. Bare trigger-word fallback

Never let a weak bare trigger word override a stronger syntactic,
compound-phrase, explicit-value, or process-context interpretation.
"""

_LOGICAL_FORM_BLOCK = """\
## PREDICATE VOCABULARY
Each predicate P(e) describes a property of a single event e (a row in event_log);
e1, e2 refer to two such events compared to each other; c refers to a case (idcase);

## Activity Predicates:
  Let A be the candidate activity reference taken from the question.

  IF A is present AND A is structurally complete, meaning A is long enough to constitute a full activity name
  AND A contains both an action word and a role word
  AND A does NOT plausibly occur as a substring within a longer activity name in the dataset
    WITH PREDICATE has_activity(e, A)
    THEN compiles to: WHERE activity = '<A>'

  IF A is present AND (A is not structurally complete as a full activity name
       OR A consists of only a single keyword, partial phrase, role-only reference, or action-only reference
       OR A could plausibly occur as a substring within a longer activity name in the dataset
       OR there is doubt about whether A qualifies as a complete activity name)
  AND this holds regardless of how the question phrases the reference to A
     (e.g. "the '<A>' activity", "associated with '<A>'", "of type '<A>'", "events involving '<A>'")
    WITH PREDICATE activity_matches(e, A)
    THEN compiles to: WHERE activity LIKE '%<A>%'

  IF the candidate activity reference A denotes a starting letter or a leading word
     (cue: "starting with", "begins with")
    WITH PREDICATE activity_starts_with(e, A)
    THEN compiles to: WHERE activity LIKE '<A>%'

  The key signal for has_activity is structural completeness of A itself, not the surrounding phrasing of the question
  If A could plausibly be a substring of a longer activity name in the dataset, treat A as a keyword and use activity_matches
  When in doubt about A's completeness, always default to activity_matches.

## Resource Predicates:
  Let R be the name of candidate resource (person) reference extracted from the question.

  IF R is present AND R refers to a specific, named person
    WITH PREDICATE performed_by(e, R)
    THEN compiles to: WHERE resource = '<R>'

  IF the question requires that some resource (any value) be assigned to the event AND no specific resource name is given
    WITH PREDICATE resource_defined(e)
    THEN compiles to: WHERE resource IS NOT NULL

  IF the question requires that no resource is assigned to the event
    WITH PREDICATE resource_undefined(e)
    THEN compiles to: WHERE resource IS NULL

## Cost Predicates:
  Let X and Y be candidate numeric values extracted from the question for cost comparisons.
  IF the question requires that a cost value exist for the event AND no specific numeric comparison is given
    WITH PREDICATE cost_known(e)
    THEN compiles to: WHERE cost IS NOT NULL

  IF the question requires that no cost value is recorded for the event
    WITH PREDICATE cost_unknown(e)
    THEN compiles to: WHERE cost IS NULL

  IF the question specifies a single numeric value X AND the comparison implies "greater than"
    WITH PREDICATE cost_gt(e, X)
    THEN compiles to: WHERE cost > X

  IF the question specifies a single numeric value X AND the comparison implies "greater than or equal to"
    WITH PREDICATE cost_gte(e, X)
    THEN compiles to: WHERE cost >= X

  IF the question specifies a single numeric value X AND the comparison implies "less than"
    WITH PREDICATE cost_lt(e, X)
    THEN compiles to: WHERE cost < X

  IF the question specifies a single numeric value X AND the comparison implies "less than or equal to"
    WITH PREDICATE cost_lt(e, X)
    THEN compiles to: WHERE cost <= X

  IF the question specifies a single numeric value X AND the comparison implies exact equality
    WITH PREDICATE cost_eq(e, X)
    THEN compiles to: WHERE cost = X

  IF the question specifies two numeric values X and Y AND the comparison implies an inclusive range between X and Y
    WITH PREDICATE cost_between(e, X, Y)
    THEN compiles to: WHERE cost BETWEEN X AND Y

## Time Predicates:
Let D, D1, D2, Y, M be candidate date/time values extracted from the question.

  IF the question references a specific date boundary D AND the comparison implies "before" that date
    WITH PREDICATE time_before(e, D)
    THEN compiles to: WHERE timestamp < 'D'

  IF the question references a specific date boundary D AND the comparison implies "after" that date
    WITH PREDICATE time_after(e, D)
    THEN compiles to: WHERE timestamp > 'D'

  IF the question references two specific date boundaries D1 and D2 AND the comparison implies an inclusive range between D1 and D2
    WITH PREDICATE time_in(e, D1, D2)
    THEN compiles to: WHERE timestamp BETWEEN 'D1' AND 'D2'

  IF the question references a year Y only AND no month or day is specified
    WITH PREDICATE in_year(e, Y)
    THEN compiles to: WHERE timestamp BETWEEN 'Y-01-01' AND 'Y+1-01-01'

  IF the question references a specific year Y and month M AND no day is specified
    WITH PREDICATE in_month(e, Y, M)
    THEN compiles to: WHERE strftime('%Y-%m', timestamp) = 'Y-MM'

  IF the question references a specific year Y, month M, and day D
    WITH PREDICATE on_day(e, Y, M, D)
    THEN compiles to: WHERE strftime('%Y-%m-%d', timestamp) = 'Y-MM-DD'

## Case ID Predicate:
Let C be the candidate case identifier extracted from the question.

  IF C is present AND C refers to a specific, known case identifier
    WITH PREDICATE in_case(e, C)
    THEN compiles to: WHERE idcase = 'C'

## Aggregation Predicates
Let S be the candidate event set, c be a candidate case, e1/e2/e be candidate events, and attr be a candidate numeric or categorical attribute, all extracted from the question.

  IF the question asks how many events there are AND no specific attribute is referenced
    WITH PREDICATE count(S)
    THEN compiles to: COUNT(*)

  IF the question asks how many different values of an attribute appear AND the question uses a cue such as "different", "distinct", or "unique"
    WITH PREDICATE distinct_count(S, attr)
    THEN compiles to: COUNT(DISTINCT attr)

  IF the question asks for a sum or total of a numeric attribute attr
    WITH PREDICATE total(S, attr)
    THEN compiles to: SUM(attr)

  IF the question asks for an average or typical value of attr
    WITH PREDICATE mean(S, attr)
    THEN compiles to: AVG(attr)

  IF the question asks for the smallest or largest value of attr AND the direction (smallest = min, largest = max) is determined by the question's wording
    WITH PREDICATE extreme(S, attr, min|max)
    THEN compiles to: MIN(attr) or MAX(attr)

  IF the question asks how long a case c took from start to finish
    WITH PREDICATE duration(c)
    THEN compiles to: strftime('%s', MAX(timestamp)) - strftime('%s', MIN(timestamp)) over events of c

  IF the question asks for the time elapsed between two specific events e1 and e2 AND e1 is the earlier event and e2 is the later event
    WITH PREDICATE time_between(e1, e2)
    THEN compiles to:  strftime('%s', e2.timestamp) - strftime('%s', e1.timestamp)

  IF the question asks how long an event e took before the next step in its case AND the case has a defined time ordering of events
    WITH PREDICATE service_duration(e)
    THEN compiles to: LEAD() window over the case ordered by time

## Events After / Before a Reference Activity
Let A be the reference activity and e the events being selected.

  IF the question asks for events / their count that occur AFTER a given activity A within the same case
    WITH PREDICATE events_after_activity(e, A)
    THEN compiles to:
      SELECT e1.<cols> FROM event_log e1
      INNER JOIN event_log e2 ON e1.idcase = e2.idcase
      WHERE e2.activity = '<A>' AND e1.timestamp > e2.timestamp
      (add GROUP BY e1.idcase + COUNT(*) when the question asks "how many")

  IF the question asks for events / their count that occur BEFORE a given activity A within the same case
    WITH PREDICATE events_before_activity(e, A)
    THEN compiles to:
      SELECT e1.<cols> FROM event_log e1
      INNER JOIN event_log e2 ON e1.idcase = e2.idcase
      WHERE e2.activity = '<A>' AND e1.timestamp < e2.timestamp

## Cross-Event Ordering Predicate:
Let A1 and A2 be candidate activity references extracted from the question, and e1, e2 be candidate events within the same case.

  IF the question asks for the number of cases in which one activity occurred before another activity
  AND A1 is the activity that must occur first
  AND A2 is the activity that must occur second
  AND has_activity(e1, A1) applies to A1 (A1 is a structurally complete activity name)
  AND has_activity(e2, A2) applies to A2 (A2 is a structurally complete activity name)
    WITH PREDICATE precedes_in_case(e1, e2, A1, A2)
    AND count(S) is applied as a DISTINCT COUNT over idcase
    THEN compiles to:
      SELECT COUNT(DISTINCT e1.idcase)
      FROM event_log e1
      INNER JOIN event_log e2 ON e1.idcase = e2.idcase
      WHERE e1.activity = '<A1>' AND e2.activity = '<A2>' AND e1.timestamp < e2.timestamp

## Elapsed Time Between Two Specific Activities
Let A1 (earlier) and A2 (later) be two specific activities in the same case.

  IF the question asks how long it took to go from activity A1 to activity A2 within a case
    WITH PREDICATE elapsed_between_activities(A1, A2)
    THEN compiles to:
      SELECT e.idcase, strftime('%s', e2.timestamp) - strftime('%s', e.timestamp)
      FROM event_log e INNER JOIN event_log e2
        ON e2.idcase = e.idcase AND e2.timestamp > e.timestamp
      WHERE e.activity = '<A1>' AND e2.activity = '<A2>'

## Superlative Predicates
Let col be the candidate grouping unit (the entity the question asks about), M be the candidate
aggregate, N be a candidate explicit number, and dir be a candidate ordering direction, all extracted from the question.

  IF the question contains a superlative
  AND no explicit number N is given
     (cue: "the most", "the highest", "the fewest", "most active", "most frequent",
      "most/least often", "most common", "occurs most")
  AND no aggregate value (COUNT, SUM, AVG, MIN, MAX) is explicitly requested
    WITH PREDICATE argmax(M) or argmin(M)
    AND ties are kept (every unit at the extreme value is returned)
    THEN compiles to:
      SELECT col
      FROM event_log
      GROUP BY col
      HAVING <agg> = (
        SELECT MAX(m)
        FROM (
          SELECT <agg> AS m
          FROM event_log
          GROUP BY col
        ) t
      );

  IF the question contains a superlative
  AND an explicit number N is given
     (cue: "top N", "the N most", "N highest", "the N least", "up to N",
      "at least N", "the N most frequent", "the N most common")
    WITH PREDICATE top(N, M) or bottom(N, M)
    AND ties at the cutoff are kept (N units at the extreme, possibly more if tied)
    THEN compiles to:
      SELECT col FROM event_log
      GROUP BY col
      HAVING <agg> IN (
        SELECT <agg> FROM event_log
        GROUP BY col ORDER BY <agg> DESC LIMIT N   -- ASC for bottom(N, M)
      );

  IF the question expresses a comparative threshold on M
     (cue: "more than N", "at least N", "fewer than N", "less than N", "exactly N")
  AND M is the aggregate being compared
    WITH PREDICATE threshold(M >= N) or threshold(M <= N)
    THEN compiles to: HAVING M >= N  (or <= N, per cue)

  IF the question contains an explicit ordering verb
  AND no other superlative, threshold, or comparative cue is present
     (cue: "ordered by X", "sorted by X", "ranked by X", "in ascending/descending order of X")
    WITH PREDICATE order_by(M, dir)
    THEN compiles to: ORDER BY M dir
    EXCEPTION: IF an explicit number N is also paired with the ordering verb
      THEN use top(N, M) instead of order_by(M, dir)

  CRITICAL: top(N, M) and "order_by + LIMIT N" are NOT interchangeable.
  IF the question says "the top N <units>"
    WITH PREDICATE top(N, M)
    THEN every unit tied at the cutoff (rank N) is preserved.
  IF the question says "list N <units>" without "top"
    WITH PREDICATE order_by(M, dir) + LIMIT N
    THEN ties at the cutoff are dropped

## First / Last Activity of a Case
Let c be a case and A a candidate activity reference.

  IF the question asks about the FIRST / initial / starting activity of each case
    WITH PREDICATE first_activity(c, A)
    THEN compiles to:
      SELECT ... FROM (SELECT idcase, activity, MIN(timestamp) FROM event_log GROUP BY idcase)
      WHERE activity = '<A>'

  IF the question asks about the LAST / final activity of each case (the activity that ended it)
    WITH PREDICATE last_activity(c, A)
    THEN compiles to:
      SELECT ... FROM (SELECT idcase, activity, MAX(timestamp) FROM event_log GROUP BY idcase)
      WHERE activity = '<A>'


## Case Handled Solely by One Specific Resource
Let R be a specific named person.

  IF the question asks for cases handled by exactly ONE resource AND that resource is the named R
    WITH PREDICATE case_solely_by(c, R)
    THEN compiles to:
      SELECT idcase FROM event_log GROUP BY idcase
      HAVING COUNT(DISTINCT resource) = 1 AND MIN(resource) = '<R>'

COUNT(DISTINCT x), MIN, MAX, SUM, AVG ignore NULL BY DESIGN, and that is the intended
  behavior. Do NOT add COUNT(*) = COUNT(x), "CASE WHEN x IS NULL", or any other NULL guard
  unless the predicate's compiled form already contains it.
DO NOT add NULL-handling logic, DO NOT EXPLORE ALTERNATIVES, EMIT IT

## Case When All Costs are NULL
  IF the question asks for cases that have NO recorded cost at all (every event's cost is NULL)
    WITH PREDICATE case_all_cost_null(c)
    THEN compiles to:
      SELECT ... FROM event_log WHERE cost IS NULL
      EXCEPT
      SELECT ... FROM event_log WHERE cost IS NOT NULL

# Negative Case Membership
  IF the question asks for cases where a given activity/condition NEVER occurs
     (cue: "have not gone through", "no record of", "did not occur",
      "still pending", "without")
    WITH PREDICATE case_lacks(cond)
    THEN compiles to:
      SELECT DISTINCT idcase FROM event_log
      WHERE idcase NOT IN (SELECT idcase FROM event_log WHERE <cond>)

# Intersection
Let cond1 and cond2 be two independent membership conditions on the same case.

  IF the question requires that BOTH conditions hold somewhere in the same case
     (cue: "both … and …", "cases involving X and also Y")
    WITH PREDICATE case_satisfies_both(cond1, cond2)
    THEN compiles to:
      SELECT idcase FROM event_log WHERE <cond1>
      INTERSECT
      SELECT idcase FROM event_log WHERE <cond2>

# Set Difference
Let cond1 (base set) and cond2 (excluded set) be membership conditions on the same column.
  IF the question asks for units satisfying cond1 but NOT cond2
     (cue: "except those…", "disregarding…", "that have not…", "not yet…",
      "X but not Y", "in A that do not occur in B")
    WITH PREDICATE case_excludes(cond1, cond2)
    THEN compiles to:
      SELECT <col> FROM event_log WHERE <cond1>
      EXCEPT
      SELECT <col> FROM event_log WHERE <cond2>

# Union of Independent Aggregates
  IF the question asks for separate counts/metrics over disjoint subsets
     reported together (cue: list of role groups, "X in year1 and Y in year2")
    WITH PREDICATE union_of(metric_1, ..., metric_n)
    THEN compiles to:
      SELECT <agg>, '<label1>' AS grp FROM event_log WHERE <cond1>
      UNION SELECT <agg>, '<label2>' ... [ORDER BY <agg>]
# Consecutive-Event Comparison (rework)
  IF the question asks for cases with the same activity performed consecutively
     (cue: "rework", "consecutively", "subsequently", "twice in a row", "same activity… again")
    WITH PREDICATE consecutive_repeat(c, attr_eq, attr_diff?)
    THEN compiles to:
      SELECT DISTINCT idcase FROM (
        SELECT idcase, activity, resource,
          LEAD(activity) OVER (PARTITION BY idcase ORDER BY timestamp) AS next_activity,
          LEAD(resource) OVER (PARTITION BY idcase ORDER BY timestamp) AS next_resource
        FROM event_log)
      WHERE activity = next_activity [AND resource != next_resource]
"""

_LOGICAL_FORM_LITE_BLOCK = """\
## ANSWER-ENTITY PREDICATES
Each predicate below classifies the answer entity of a question Q: the
thing Q asks to return, count, list, compare, or aggregate. Exactly one of
is_event, is_case, is_activity, is_resource holds for a well-formed Q. Let
H be the syntactic head noun of Q's answer entity.

IF H is one of: event, occurrence, performance, instance, record, moment,
   row, line, tuple, intervention
  OR H is "action" referring to a single execution or occurrence (not a
     type or category)
  AND no COMPOUND-PHRASE PREDICATE below overrides H
    WITH PREDICATE is_event(Q)
    THEN compiles to:
      SELECT <event-level column(s)>
      FROM event_log
      [WHERE ...]
    Do NOT GROUP BY idcase, activity, or resource unless Q explicitly asks
    for a per-case, per-activity, or per-resource breakdown.

IF H is one of: case, process instance, declaration, travel declaration,
   permit, process
  OR H is a bare "trip" with no start/end/beginning/finalization wording
     attached (see COMPOUND-PHRASE PREDICATES)
  AND no COMPOUND-PHRASE PREDICATE below overrides H
    WITH PREDICATE is_case(Q)
    THEN compiles to:
      SELECT [DISTINCT] idcase
      FROM event_log
      [WHERE ...]
      [GROUP BY idcase [HAVING ...]]
    Use GROUP BY idcase when the condition depends on a property of the
    whole case, such as a count, minimum, maximum, or sum over that
    case's events.

IF H is one of: activity, activities, task, action-type
  OR H is "action" referring to an activity type, category, or kind of
     action (not a single execution)
  OR Q contains "different" or "distinct" immediately before a plural
     activity-like noun (e.g. "how many different activities")
    WITH PREDICATE is_activity(Q)
    THEN compiles to:
      SELECT [DISTINCT] activity
      FROM event_log
      [WHERE ...]
      [GROUP BY activity [HAVING ...]]
    Use GROUP BY activity when Q asks for a metric broken down by
    activity type, such as a count per activity.

    Note: an activity value V need not contain the word "activity" to
    satisfy has_activity(e, V) / activity_matches(e, V) elsewhere in a
    compiled query -- e.g. 'Start trip', 'End trip', 'Send reminder',
    'Rejected by supervisor' are all activity values. Their presence in Q
    does NOT by itself satisfy is_activity(Q); see NAMED-VALUE PREDICATE.

    Examples:
      "How many events occurred for 'Start trip'?"
        H = "events" -> is_event(Q); is_filter_value('Start trip')
      "Which activities include 'Start trip'?"
        H = "activities" -> is_activity(Q)
      "How many different activities were performed?"
        H = "activities" + "different" modifier -> is_activity(Q)

IF H is one of: resource, employee, person, people, collaborator
  OR H is a role name used as the answer entity: supervisor, director,
     administration, administrator, pre-approver, approver, budget owner,
     budget holder
    WITH PREDICATE is_resource(Q)
    THEN compiles to:
      SELECT [DISTINCT] resource
      FROM event_log
      [WHERE ...]
      [GROUP BY resource [HAVING ...]]
    Use GROUP BY resource when Q asks for a metric broken down by
    resource, such as a count per employee.

    Note: a role name appearing inside a named activity value does NOT by
    itself satisfy is_resource(Q) -- it may be part of an activity string
    instead. Check whether the role name is Q's own head noun H, or
    merely a substring of a quoted/named activity value.

    Examples:
      "Which employees performed the events?"
        H = "employees" -> is_resource(Q)
      "How many events were performed by supervisors?"
        H = "events"; "supervisors" is a resource-side qualifier, not
        necessarily H itself -> is_event(Q) with a resource condition, or
        is_resource(Q), depending on which noun Q actually asks to return
      "How many 'Rejected by supervisor' activities occurred?"
        H = "activities" -> is_activity(Q); 'supervisor' here is part of
        the quoted activity value, not a resource reference

## COMPOUND-PHRASE PREDICATES
Let A be a candidate compound phrase in Q. A compound-phrase match
overrides the bare-trigger-word predicates above.

IF A = "process instance"
    WITH PREDICATE compound_overrides(A, is_case)
    THEN answer_entity(Q) = is_case(Q), even though bare "instance" alone
    would satisfy is_event(Q).

IF A matches start/end/beginning/finalization + "trip"
   (e.g. "start of a trip", "end of a trip", "beginning of the trip",
   "finalization of a trip")
    WITH PREDICATE compound_overrides(A, is_activity)
    THEN A denotes the Activity value 'Start trip' or 'End trip', not the
    case itself:
      answer_entity(Q) = is_activity(Q), or a filter
      has_activity(e, 'Start trip'|'End trip'), never is_case(Q).
    Only a bare "trip" with none of this wording attached may satisfy
    is_case(Q).

IF A is "declaration(s)" or "permit(s)" occurring inside a
   PROCESS-CONTEXT PREDICATE below
    WITH PREDICATE compound_overrides(A, is_process_context)
    THEN A does not satisfy is_case(Q) and does not create a filter; it
    only satisfies is_case(Q) when it is itself Q's answer entity, outside
    any process-context phrase.

## NAMED-VALUE PREDICATE
Let V be a quoted or named activity/resource value appearing in Q.

IF V is present
  AND Q's answer-entity predicate (from ANSWER-ENTITY PREDICATES) is
      already satisfied by a different head noun H != V
    WITH PREDICATE is_filter_value(V), NOT answer_entity(V)
    THEN V compiles to a filter on the corresponding column
      (activity = '<V>' or resource = '<V>'), and does NOT change the
      answer entity determined by H.
    Example: "How many events occurred for the activity 'Start trip'?"
      answer_entity(Q) = is_event(Q)   [H = "events"]
      is_filter_value('Start trip')     -> WHERE activity = 'Start trip'

## PROCESS-CONTEXT PREDICATE
Let P be a prepositional construction in Q of the form:
  <preposition: during | in | within | throughout | across | for | when |
   while | of | to | over the course of> + declaration/process/processing
   wording
  (e.g. "during the processing of declarations", "throughout the
  declaration process", "across the event log", "to process the
  declarations")

IF Q contains P
    WITH PREDICATE is_process_context(P)
    THEN P contributes NO activity or case filter and does NOT change
    answer_entity(Q):
      Do NOT translate P into activity = ... or activity LIKE ...
    unless Q separately, outside P, identifies a concrete activity value
    or explicitly asks for activities matching that value.
    Example: "How many events occurred during the processing of
    declarations?"
      answer_entity(Q) = is_event(Q)   [H = "events"]
      is_process_context("during the processing of declarations")
      -> no activity filter

## RESOLUTION-ORDER PREDICATE
IF more than one predicate above matches Q
    WITH PREDICATE resolve(Q)
    THEN prefer, in this order:
      1. An explicit answer-entity head noun (ANSWER-ENTITY PREDICATES)
      2. An explicit named/quoted filter value (NAMED-VALUE PREDICATE)
      3. A compound-phrase override (COMPOUND-PHRASE PREDICATES)
      4. The syntactic head noun alone
      5. A recognized process-context construction (PROCESS-CONTEXT
         PREDICATE)
      6. A bare trigger word, as a last resort
    Never let a weak bare trigger word override a stronger compound-
    phrase, named-value, or process-context predicate above it in this
    order.
"""

PURE_ZEROSHOT_SYSTEM = f"{_BASE_HEADER}\n{_FOOTER}"
LEXICON_ZEROSHOT_SYSTEM = f"{_BASE_HEADER}\n{_LEXICON_BLOCK}\n{_FOOTER}"
LOGICAL_FORM_SYSTEM = f"{_BASE_HEADER}\n{_LOGICAL_FORM_BLOCK}\n{_FOOTER}"
LOGICAL_FORM_LITE_SYSTEM = f"{_BASE_HEADER}\n{_LOGICAL_FORM_LITE_BLOCK}\n{_FOOTER}"
LEXICON_LOGICAL_FORM_SYSTEM = (
    f"{_BASE_HEADER}\n{_LEXICON_BLOCK}\n{_LOGICAL_FORM_BLOCK}\n{_FOOTER}"
)


# ---------------------------------------------------------------------------
class PureZeroShot(PromptStrategy):
    """Pure zero-shot: system instruction + bare question."""

    name = "0-shot-pure"

    def _system_message(self) -> ChatMessage:
        return ChatMessage(role="system", content=PURE_ZEROSHOT_SYSTEM)

    def build(self, sample: Sample) -> list[ChatMessage]:
        user = f"Question: {sample.question}"
        return [self._system_message(), ChatMessage(role="user", content=user)]


class LexiconGuided(PromptStrategy):
    """Zero-shot with a domain lexicon block in the system prompt."""

    name = "0-shot-lexicon"

    def _system_message(self) -> ChatMessage:
        return ChatMessage(role="system", content=LEXICON_ZEROSHOT_SYSTEM)

    def build(self, sample: Sample) -> list[ChatMessage]:
        user = f"Question: {sample.question}"
        return [self._system_message(), ChatMessage(role="user", content=user)]


class LogicalFormGuided(PromptStrategy):
    """Zero-shot guided by an intermediate logical-form sketch."""

    name = "lf"

    def _system_message(self) -> ChatMessage:
        return ChatMessage(role="system", content=LOGICAL_FORM_SYSTEM)

    def build(self, sample: Sample) -> list[ChatMessage]:
        user = f"Question: {sample.question}"
        return [self._system_message(), ChatMessage(role="user", content=user)]


class LogicalFormLiteGuided(PromptStrategy):
    """Zero-shot guided by a logical-form-style block scoped to ONLY answer-
    entity/target-column selection -- the same job ``_LEXICON_BLOCK`` does,
    expressed in the ``lf`` block's IF/WITH PREDICATE/THEN formalism instead
    of trigger-word prose. Exists to pair against ``0-shot-lexicon`` as a
    scope-matched ablation: both blocks answer the identical sub-problem,
    so a delta between them isolates guidance *style*, not guidance
    *scope* (unlike ``lf``, which also covers filters/aggregation/joins/
    time logic that ``_LEXICON_BLOCK`` never touches).
    """

    name = "lf-lite"

    def _system_message(self) -> ChatMessage:
        return ChatMessage(role="system", content=LOGICAL_FORM_LITE_SYSTEM)

    def build(self, sample: Sample) -> list[ChatMessage]:
        user = f"Question: {sample.question}"
        return [self._system_message(), ChatMessage(role="user", content=user)]


class LexiconLogicalFormGuided(PromptStrategy):
    """Zero-shot guided by BOTH the domain lexicon and the full logical-form
    block, back to back in the same system prompt. Fourth cell of the 2x2
    {lexicon} x {lf} factorial -- pairs with ``PureZeroShot``, ``ZeroShotLexicon``,
    and ``LogicalFormGuided`` so a lexicon:lf interaction effect can be
    estimated, not just the two main effects.
    """

    name = "lexicon-lf"

    def _system_message(self) -> ChatMessage:
        return ChatMessage(role="system", content=LEXICON_LOGICAL_FORM_SYSTEM)

    def build(self, sample: Sample) -> list[ChatMessage]:
        user = f"Question: {sample.question}"
        return [self._system_message(), ChatMessage(role="user", content=user)]
