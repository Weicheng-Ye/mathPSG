"""Dimension-independent words and graded group presentations.

The benchmark modules historically defined their own ``Word`` aliases and
small manipulation helpers.  Classifier v0.1 uses the common definitions in
this module.  Existing benchmark words are structurally compatible with this
representation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Generic, Mapping, TypeVar


Letter = tuple[str, int]
Word = tuple[Letter, ...]


def inverse_word(word: Word) -> Word:
    """Return the formal inverse of ``word``."""

    return tuple(
        (generator, -exponent)
        for generator, exponent in reversed(word)
        if exponent
    )


def word_power(word: Word, exponent: int) -> Word:
    """Return a formal integer power."""

    if exponent < 0:
        return word_power(inverse_word(word), -exponent)
    return word * exponent


def substitute_word(
    word: Word,
    substitutions: Mapping[str, Word],
) -> Word:
    """Substitute a word for every named generator."""

    result: Word = ()
    for generator, exponent in word:
        replacement = substitutions[generator]
        result += word_power(replacement, exponent)
    return result


Element = TypeVar("Element")


def evaluate_word(
    word: Word,
    generators: Mapping[str, Element],
    identity: Element,
    *,
    multiply: Callable[[Element, Element], Element],
    inverse: Callable[[Element], Element],
) -> Element:
    """Evaluate ``word`` with explicit multiplication and inverse maps."""

    result = identity
    for generator, exponent in word:
        value = generators[generator]
        if exponent < 0:
            value = inverse(value)
        for _ in range(abs(exponent)):
            result = multiply(result, value)
    return result


@dataclass(frozen=True)
class GradedPresentation:
    """A finite presentation together with a ``Z2`` generator grading."""

    generators: tuple[str, ...]
    relators: Mapping[str, Word]
    grades: Mapping[str, int]

    def __post_init__(self) -> None:
        if len(set(self.generators)) != len(self.generators):
            raise ValueError("presentation generators must be unique")
        generator_set = set(self.generators)
        if set(self.grades) != generator_set:
            raise ValueError("every generator must have exactly one grade")
        for label, word in self.relators.items():
            unknown = {
                generator
                for generator, _ in word
                if generator not in generator_set
            }
            if unknown:
                raise ValueError(
                    f"relator {label!r} uses unknown generators "
                    f"{sorted(unknown)!r}"
                )
            if self.word_grade(word):
                raise ValueError(
                    f"relator {label!r} has nonzero antiunitary grade"
                )

    def word_grade(self, word: Word) -> int:
        return sum(
            self.grades[generator] * exponent
            for generator, exponent in word
        ) % 2


Value = TypeVar("Value")


@dataclass(frozen=True)
class NamedWordValue(Generic[Value]):
    """A relator label paired with an evaluated value."""

    label: str
    value: Value
