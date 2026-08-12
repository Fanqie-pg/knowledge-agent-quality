from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from app.query_rewriter import QueryRewriter


class FakeRewriteModel:
    def __init__(self, rewritten_query):
        self.rewritten_query = rewritten_query
        self.last_messages = None

    def invoke(self, messages):
        self.last_messages = messages
        return AIMessage(content=self.rewritten_query)


def test_query_rewriter_returns_model_content():
    model = FakeRewriteModel(
        "账号连续输入错误密码几次会被锁定，多久自动解锁？"
    )

    rewriter = QueryRewriter(model)

    result = rewriter.rewrite(
        "登录口令连续写错多少回会被封？多长时间后自己恢复？"
    )

    assert result == "账号连续输入错误密码几次会被锁定，多久自动解锁？"
    assert len(model.last_messages) == 2
    assert isinstance(model.last_messages[0], SystemMessage)
    assert isinstance(model.last_messages[1], HumanMessage)
    assert model.last_messages[1].content == "登录口令连续写错多少回会被封？多长时间后自己恢复？"