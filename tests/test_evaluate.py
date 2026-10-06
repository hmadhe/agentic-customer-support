from langchain_core.messages import AIMessage, HumanMessage

from app.ingest import POLICIES_DIR
from app.schemas import Intent, IntentClassification, PolicyAnswer, Sentiment
from scripts.evaluate import EvalCase, check_case, load_cases, route_of


def result(intent=Intent.POLICY_QUESTION, reply="Express shipping costs $14.99.", sources=("shipping.md",),
           escalation_reason=None, tool_calls=()):
    messages = [HumanMessage("question")]
    if tool_calls:
        messages.append(AIMessage("", tool_calls=[{"name": name, "args": {}, "id": name} for name in tool_calls]))
    return {
        "classification": IntentClassification(intent=intent, sentiment=Sentiment.NEUTRAL),
        "policy_answer": PolicyAnswer(answered=True, answer=reply, sources=list(sources)) if sources else None,
        "escalation_reason": escalation_reason,
        "messages": messages,
        "response": reply,
    }


def test_golden_set_is_valid():
    cases = load_cases()
    policy_files = {path.name for path in POLICIES_DIR.glob("*.md")}

    assert len(cases) >= 40
    assert len({case.id for case in cases}) == len(cases)
    assert all(case.turns and case.source in (None, *policy_files) for case in cases)
    # "no" or a single letter as an accepted answer would match almost any reply.
    assert all(len(option) >= 2 and option.lower() != "no" for case in cases for group in case.must_contain for option in group.split("|"))


def test_route_is_derived_from_escalation_first_then_intent():
    assert route_of(result()) == "rag"
    assert route_of(result(intent=Intent.ORDER_ISSUE)) == "agent"
    assert route_of(result(intent=Intent.GREETING)) == "respond"
    assert route_of(result(escalation_reason="policy_not_found")) == "escalate"


def test_matching_case_passes_every_check():
    case = EvalCase(id="x", category="policy", turns=["q"], route="rag", source="shipping.md", must_contain=["14.99|fourteen"])

    assert check_case(case, result()) == {"route": True, "source": True, "facts": True}


def test_each_check_can_fail_on_its_own():
    case = EvalCase(
        id="x", category="order", turns=["q"], route="agent", escalation_reason="tool_error", source="returns.md",
        tools=["check_return_eligibility"], must_contain=["15%"], must_not_contain=["$14.99"],
    )

    checks = check_case(case, result(tool_calls=["get_order_status"]))

    assert checks == {
        "route": False, "escalation_reason": False, "source": False, "tools": False, "facts": False, "no_hallucination": False,
    }


def test_no_tools_and_case_insensitive_facts():
    case = EvalCase(id="x", category="order", turns=["q"], no_tools=True, must_contain=["ORDER NUMBER|order id"])

    asked = result(intent=Intent.ORDER_ISSUE, reply="Could you give me your order number?", sources=())
    looked_up = result(intent=Intent.ORDER_ISSUE, reply="Your order number is 1", sources=(), tool_calls=["get_order_status"])

    assert check_case(case, asked) == {"no_tools": True, "facts": True}
    assert check_case(case, looked_up)["no_tools"] is False
