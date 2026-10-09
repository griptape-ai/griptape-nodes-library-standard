from griptape.chunkers import TextChunker
from griptape.tokenizers import SimpleTokenizer
from griptape_nodes.exe_types.core_types import Parameter, ParameterMode
from griptape_nodes.exe_types.node_types import AsyncResult

from griptape_nodes_library.tasks.base_task import BaseTask

DEFAULT_MODEL = "gpt-4.1-nano"

# Tokens per request, under every offered model's context. Longer text is summarized
# in chunks, then the joined summaries are summarized.
MAX_CHUNK_TOKENS = 100_000
CHARS_PER_TOKEN = 4

# The prompts griptape's PromptSummaryEngine used.
SUMMARY_INSTRUCTIONS = "You are an expert in text summarization."
SUMMARY_USER_TEMPLATE = '''Summarize the following text: """
{text}
"""

Summary:'''


class SummarizeText(BaseTask):
    """Base task node for creating Griptape Tasks that can run on their own.

    Attributes:
        prompt (BaseTool): A dictionary representation of the created tool.
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.add_parameter(
            Parameter(
                name="prompt",
                input_types=["str"],
                type="str",
                output_type="str",
                default_value="",
                tooltip="",
                ui_options={"multiline": True, "placeholder_text": "Input text to process"},
            )
        )
        self._add_model_parameter(default_model=DEFAULT_MODEL)

        self.add_parameter(
            Parameter(
                name="output",
                type="str",
                output_type="str",
                default_value=None,
                allowed_modes={ParameterMode.OUTPUT},
                tooltip="The output of the task.",
                ui_options={"multiline": True, "placeholder_text": "Task output"},
            )
        )

    def process(self) -> AsyncResult[str]:
        model = self._require_permitted_model()

        text = self.get_parameter_value("prompt")
        if text and not text.isspace():
            yield lambda: self._summarize(text, model)

    def _summarize(self, text: str, model: str) -> str:
        tokenizer = SimpleTokenizer(characters_per_token=CHARS_PER_TOKEN)
        if tokenizer.count_tokens(text) <= MAX_CHUNK_TOKENS:
            user_prompt = SUMMARY_USER_TEMPLATE.format(text=text)
            return self._process(user_prompt, model, instructions=SUMMARY_INSTRUCTIONS).text
        summaries = [
            self._process(
                SUMMARY_USER_TEMPLATE.format(text=chunk.value),
                model,
                instructions=SUMMARY_INSTRUCTIONS,
                stream_output=False,
            ).text
            for chunk in TextChunker(tokenizer=tokenizer, max_tokens=MAX_CHUNK_TOKENS).chunk(text)
        ]
        return self._summarize("\n\n".join(summaries), model)
