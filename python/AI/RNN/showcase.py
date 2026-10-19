"""
Shared output helpers for the showcase scripts (the main.py in each block).

Usage in a block's main.py:

    from AI.RNN.showcase import Showcase, header, takeaway

    showcase = Showcase()

    @showcase.experiment
    def my_experiment(ctx):
        '''Title on the first line

        Introduction, printed before the output.
        '''
        ...

    showcase.run(ctx)

Only the showcase scripts should import this file. The real code
(tokenizer.py, embeddings.py, rnn.py, ...) should not depend on it.
"""
import inspect
import os
import sys

WIDTH = 60


def style(text: str, code: str) -> str:
    """Wraps text in an ANSI style code (plain text if piped to a file)."""
    use_color = sys.stdout.isatty() or "PYCHARM_HOSTED" in os.environ
    return f"\033[{code}m{text}\033[0m" if use_color else text


def header(title: str) -> None:
    print("\n\n\n" + style("=" * WIDTH, "36"))
    print(style(title, "1;36"))  # bold cyan
    print(style("=" * WIDTH, "36") + "\n")


def takeaway(*lines: str) -> None:
    """Prints the key point(s) of an experiment after its output."""
    print()
    for line in lines:
        print(style(f"-> {line}", "32"))  # green


def bolden(text: str) -> str:
    """Returns the text in bold ANSI style (plain text if piped to a file)."""
    return style(text, "1")


class Showcase:
    """Collects experiments and runs them in definition order.

    Each experiment's docstring is printed as its introduction:
        First line  -> experiment title
        Blank line
        Rest        -> short introduction
    """

    def __init__(self) -> None:
        self.experiments = []

    def experiment(self, fn):
        """Decorator that registers a function as an experiment."""
        self.experiments.append(fn)
        return fn

    def run(self, ctx) -> None:
        for number, fn in enumerate(self.experiments, start=1):
            title, _, description = inspect.getdoc(fn).partition("\n\n")
            header(f"EXPERIMENT {number}: {title}")
            print(style(description, "2") + "\n")  # dim
            fn(ctx)
