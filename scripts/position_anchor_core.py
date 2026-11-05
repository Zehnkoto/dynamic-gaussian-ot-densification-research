"""Uniform support anchor; keeps the original source CTD field for evaluation."""
def uniform_relative_penalty(field):
    return field.offset().double().square().sum(1).mean()/field.baseline
